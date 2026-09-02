package com.mangomamba.benchmark

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.Debug
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.roundToLong
import kotlin.random.Random

class MainActivity : AppCompatActivity() {

    private val modelAssetName = "mango_mamba_7class.onnx"
    private val inputName = "image"
    private val imageSize = 224
    private val resizeShortSide = 256
    private val warmupRuns = 5
    private val timedRuns = 50
    private val testImageExtra = "test_image_path"
    private val labels = listOf(
        "Anthracnose",
        "Dieback",
        "Gall Midge",
        "Healthy",
        "Powdery Mildew",
        "Red Rust",
        "Sooty Mold"
    )

    private lateinit var statusText: TextView
    private lateinit var resultText: TextView
    private lateinit var logFile: File
    private lateinit var classificationLogFile: File
    companion object {
        private var sharedInferenceSession: OrtSession? = null
        private val sessionLock = Any()
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        resultText = findViewById(R.id.resultText)
        logFile = File(getExternalFilesDir(null), "benchmark_log.txt")
        classificationLogFile = File(getExternalFilesDir(null), "classification_log.txt")

        findViewById<Button>(R.id.runButton).setOnClickListener {
            lifecycleScope.launch { runBenchmark() }
        }
        findViewById<Button>(R.id.classifyButton).setOnClickListener {
            startActivityForResult(
                Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "image/*"
                },
                100
            )
        }

        handleTestImageIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleTestImageIntent(intent)
    }


    @Deprecated("Used for the simple dev-mode image picker")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 100 && resultCode == RESULT_OK) {
            data?.data?.let { classifyUri(it) }
        }
    }

    private fun handleTestImageIntent(intent: Intent?) {
        intent?.getStringExtra(testImageExtra)?.takeIf { it.isNotBlank() }?.let { classifyPath(it) }
    }

    private fun classifyUri(uri: Uri) {
        lifecycleScope.launch {
            setStatus("Đang phân loại ảnh...")
            val report = withContext(Dispatchers.Default) {
                runCatching {
                    contentResolver.openInputStream(uri).use { input ->
                        val bitmap = BitmapFactory.decodeStream(input)
                            ?: error("Không đọc được ảnh")
                        classifyBitmap(bitmap, uri.toString())
                    }
                }.getOrElse { errorReport(it) }
            }
            appendToLogFile(classificationLogFile, report)
            resultText.text = report
            setStatus("Xong. Kết quả đã lưu vào ${classificationLogFile.absolutePath}")
        }
    }

    private fun classifyPath(path: String) {
        lifecycleScope.launch {
            setStatus("Đang phân loại $path ...")
            val report = withContext(Dispatchers.Default) {
                runCatching {
                    val bitmap = BitmapFactory.decodeFile(path) ?: error("Không đọc được ảnh: $path")
                    classifyBitmap(bitmap, path)
                }.getOrElse { errorReport(it) }
            }
            appendToLogFile(classificationLogFile, report)
            resultText.text = report
            setStatus("Xong. Kết quả đã lưu vào ${classificationLogFile.absolutePath}")
        }
    }

    private fun classifyBitmap(bitmap: Bitmap, source: String): String {
        val env = OrtEnvironment.getEnvironment()
        val session = getInferenceSession(env)
        val tensorData = bitmapToTensor(bitmap)
        val tensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(tensorData),
            longArrayOf(1, 3, imageSize.toLong(), imageSize.toLong())
        )
        val startNanos = System.nanoTime()
        val result = session.run(mapOf(inputName to tensor))
        val elapsedMs = (System.nanoTime() - startNanos) / 1_000_000.0
        val rawOutput = result[0].value
        val logits = when (rawOutput) {
            is FloatArray -> rawOutput
            is Array<*> -> rawOutput[0] as FloatArray
            else -> error("Output không hỗ trợ: ${rawOutput::class.java}")
        }
        val probabilities = softmax(logits)
        val ranking = probabilities.indices.sortedByDescending { probabilities[it] }

        val sb = StringBuilder()
        sb.appendLine("Source: $source")
        sb.appendLine("Input: ${bitmap.width}x${bitmap.height} -> short side $resizeShortSide -> center crop $imageSize")
        sb.appendLine("Inference: ${"%.1f".format(elapsedMs)} ms")
        sb.appendLine()
        sb.appendLine("Top-3:")
        ranking.take(3).forEachIndexed { index, classIndex ->
            sb.appendLine("${index + 1}. ${labels[classIndex]}: ${"%.2f".format(probabilities[classIndex] * 100)}%")
        }

        result.close()
        tensor.close()
        return sb.toString()
    }

    private fun getInferenceSession(env: OrtEnvironment): OrtSession {
        return sharedInferenceSession ?: synchronized(sessionLock) {
            sharedInferenceSession ?: run {
                val modelBytes = assets.open(modelAssetName).use { it.readBytes() }
                val options = OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                }
                env.createSession(modelBytes, options).also { sharedInferenceSession = it }
            }
        }
    }

    private fun bitmapToTensor(bitmap: Bitmap): FloatArray {
        val scale = resizeShortSide.toFloat() / min(bitmap.width, bitmap.height)
        val resizedWidth = (bitmap.width * scale).toDouble().roundToLong().toInt()
        val resizedHeight = (bitmap.height * scale).toDouble().roundToLong().toInt()
        val resized = Bitmap.createScaledBitmap(bitmap, resizedWidth, resizedHeight, true)
        val left = (resizedWidth - imageSize) / 2
        val top = (resizedHeight - imageSize) / 2
        val cropped = Bitmap.createBitmap(resized, left, top, imageSize, imageSize)
        val tensor = FloatArray(3 * imageSize * imageSize)
        val means = floatArrayOf(0.485f, 0.456f, 0.406f)
        val stds = floatArrayOf(0.229f, 0.224f, 0.225f)
        for (y in 0 until imageSize) {
            for (x in 0 until imageSize) {
                val color = cropped.getPixel(x, y)
                val offset = y * imageSize + x
                tensor[offset] = ((android.graphics.Color.red(color) / 255f) - means[0]) / stds[0]
                tensor[imageSize * imageSize + offset] =
                    ((android.graphics.Color.green(color) / 255f) - means[1]) / stds[1]
                tensor[2 * imageSize * imageSize + offset] =
                    ((android.graphics.Color.blue(color) / 255f) - means[2]) / stds[2]
            }
        }
        if (resized !== cropped) resized.recycle()
        cropped.recycle()
        return tensor
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val exponentials = FloatArray(logits.size) { exp((logits[it] - maxLogit).toDouble()).toFloat() }
        val sum = exponentials.sum()
        return FloatArray(logits.size) { exponentials[it] / sum }
    }

    private suspend fun runBenchmark() {
        setStatus("Đang chạy benchmark...")
        val report = withContext(Dispatchers.Default) {
            try {
                benchmarkOnce()
            } catch (t: Throwable) {
                Log.e("Phase0Bench", "Benchmark failed", t)
                errorReport(t)
            }
        }
        appendToLogFile(logFile, report)
        resultText.text = report
        setStatus("Xong. Kết quả đã lưu vào ${logFile.absolutePath}")
    }

    private fun benchmarkOnce(): String {
        val sb = StringBuilder()
        val pssBefore = currentPssKb()
        val modelBytes = assets.open(modelAssetName).use { it.readBytes() }
        sb.appendLine("Model asset size: ${modelBytes.size / (1024 * 1024)} MB")

        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions()
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

        val cpuCores = Runtime.getRuntime().availableProcessors()
        val intraThreads = maxOf(1, cpuCores / 2)
        opts.setIntraOpNumThreads(intraThreads)
        sb.appendLine("CPU cores: $cpuCores, intraOpNumThreads: $intraThreads")

        var xnnpackEnabled = false
        try {
            opts.addXnnpack(mapOf("intra_op_num_threads" to intraThreads.toString()))
            xnnpackEnabled = true
        } catch (t: Throwable) {
            Log.w("Phase0Bench", "XNNPACK EP unavailable, falling back to pure CPU EP", t)
        }
        sb.appendLine("XNNPACK EP enabled: $xnnpackEnabled")

        val initStartNanos = System.nanoTime()
        val session = env.createSession(modelBytes, opts)
        val initMs = (System.nanoTime() - initStartNanos) / 1_000_000.0
        sb.appendLine("Session init time: ${"%.1f".format(initMs)} ms")
        sb.appendLine("Graph inputs: ${session.inputNames} outputs: ${session.outputNames}")

        val pssAfterInit = currentPssKb()
        val rnd = Random(42)
        val floatArr = FloatArray(1 * 3 * imageSize * imageSize) { rnd.nextFloat() * 2f - 1f }

        fun runOnce(): Double {
            val tensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(floatArr),
                longArrayOf(1, 3, imageSize.toLong(), imageSize.toLong())
            )
            val t0 = System.nanoTime()
            val result = session.run(mapOf(inputName to tensor))
            val elapsedMs = (System.nanoTime() - t0) / 1_000_000.0
            result.close()
            tensor.close()
            return elapsedMs
        }

        val warmupTimes = DoubleArray(warmupRuns)
        for (i in 0 until warmupRuns) warmupTimes[i] = runOnce()
        sb.appendLine("Warm-up runs (ms): ${warmupTimes.joinToString(", ") { "%.1f".format(it) }}")

        val times = DoubleArray(timedRuns)
        for (i in 0 until timedRuns) times[i] = runOnce()
        times.sort()
        val pssAfterRun = currentPssKb()

        fun percentile(p: Double): Double {
            val idx = ((p / 100.0) * (times.size - 1)).roundToLong().toInt()
            return times[idx.coerceIn(0, times.size - 1)]
        }

        sb.appendLine()
        sb.appendLine("=== Steady-state latency over $timedRuns runs (ms) ===")
        sb.appendLine("min   : ${"%.2f".format(times.first())}")
        sb.appendLine("p50   : ${"%.2f".format(percentile(50.0))}")
        sb.appendLine("p90   : ${"%.2f".format(percentile(90.0))}")
        sb.appendLine("p95   : ${"%.2f".format(percentile(95.0))}")
        sb.appendLine("p99   : ${"%.2f".format(percentile(99.0))}")
        sb.appendLine("max   : ${"%.2f".format(times.last())}")
        sb.appendLine("mean  : ${"%.2f".format(times.average())}")
        sb.appendLine()
        sb.appendLine("=== Memory (PSS, MB) ===")
        sb.appendLine("Before load   : ${pssBefore / 1024.0}")
        sb.appendLine("After session init : ${pssAfterInit / 1024.0}")
        sb.appendLine("After $timedRuns runs : ${pssAfterRun / 1024.0}")
        sb.appendLine()
        sb.appendLine("=== Phase 0 gate (see plan.md §3.2) ===")
        val p95 = percentile(95.0)
        val verdict = when {
            p95 <= 120.0 -> "GO — p95 <= 120ms, tiếp tục bình thường."
            p95 <= 400.0 -> "GO WITH OPTIMIZATION — 120-400ms, áp dụng §3.4 a/b/c trước khi build UI đầy đủ."
            else -> "STOP — p95 > 400ms, cần xử lý §3.4 d/e trước khi tiếp tục."
        }
        sb.appendLine("p95 = ${"%.1f".format(p95)} ms -> $verdict")
        session.close()
        return sb.toString()
    }

    private fun errorReport(t: Throwable): String =
        "LỖI: ${t::class.java.simpleName}: ${t.message}\n\n${Log.getStackTraceString(t)}"

    private fun currentPssKb(): Int = Debug.getPss().toInt()

    private fun setStatus(text: String) {
        runOnUiThread { statusText.text = text }
    }

    private fun appendToLogFile(file: File, text: String) {
        try {
            FileOutputStream(file, true).use { out ->
                out.write("\n----- ${java.util.Date()} -----\n".toByteArray())
                out.write(text.toByteArray())
                out.write("\n".toByteArray())
            }
        } catch (t: Throwable) {
            Log.e("MangoMamba", "Failed to write log file", t)
        }
    }
}
