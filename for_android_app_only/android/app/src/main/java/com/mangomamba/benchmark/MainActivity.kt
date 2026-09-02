package com.mangomamba.benchmark

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.Surface
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.lifecycleScope
import com.google.android.material.button.MaterialButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.FloatBuffer
import java.util.Locale
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.roundToLong

class MainActivity : AppCompatActivity() {

    private val modelAssetName = "mango_mamba_7class.onnx"
    private val labelsAssetName = "mango_mamba_7class_labels.json"
    private val inputName = "image"
    private val imageSize = 224
    private val resizeShortSide = 256
    private val testImageExtra = "test_image_path"
    private val captureIntervalMs = 800L
    private val sourceUrl = Uri.parse(
        "https://sansangxuatkhau.ppd.gov.vn/kien-thuc-kinh-nghiem/" +
            "hai-giai-phap-sinh-hoc-hieu-qua-o-vung-xoai-cam-lam.html"
    )

    private lateinit var cameraPreview: PreviewView
    private lateinit var cameraScreen: LinearLayout
    private lateinit var informationScreen: LinearLayout
    private lateinit var statusText: TextView
    private lateinit var resultImage: ImageView
    private lateinit var resultName: TextView
    private lateinit var resultConfidence: TextView
    private lateinit var resultMessage: TextView
    private lateinit var captureButton: MaterialButton
    private lateinit var continuousButton: MaterialButton
    private lateinit var informationButton: MaterialButton
    private lateinit var informationTitle: TextView
    private lateinit var informationSubtitle: TextView
    private lateinit var symptomsText: TextView
    private lateinit var preventionText: TextView
    private lateinit var helpText: TextView
    private lateinit var referenceButton: MaterialButton

    private val labels by lazy { loadLabels() }
    private val mainHandler = Handler(Looper.getMainLooper())
    private var imageCapture: ImageCapture? = null
    private var isCaptureOrInferenceRunning = false
    private var continuousCaptureEnabled = false
    private var latestPrediction: Prediction? = null

    private val continuousRunnable = object : Runnable {
        override fun run() {
            if (!continuousCaptureEnabled) return
            if (!isCaptureOrInferenceRunning) capturePhoto()
            mainHandler.postDelayed(this, captureIntervalMs)
        }
    }

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startCamera()
        } else {
            setStatus(getString(R.string.camera_permission_needed))
        }
    }

    companion object {
        private var sharedInferenceSession: OrtSession? = null
        private val sessionLock = Any()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        bindViews()
        bindActions()
        requestCameraWhenNeeded()
        handleTestImageIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleTestImageIntent(intent)
    }

    override fun onPause() {
        super.onPause()
        stopContinuousCapture()
    }

    override fun onDestroy() {
        mainHandler.removeCallbacks(continuousRunnable)
        if (isFinishing) {
            synchronized(sessionLock) {
                sharedInferenceSession?.close()
                sharedInferenceSession = null
            }
        }
        super.onDestroy()
    }

    @Deprecated("Back navigation for the embedded information screen")
    override fun onBackPressed() {
        if (informationScreen.visibility == View.VISIBLE) {
            showCameraScreen()
        } else {
            super.onBackPressed()
        }
    }

    private fun bindViews() {
        cameraPreview = findViewById(R.id.cameraPreview)
        cameraScreen = findViewById(R.id.cameraScreen)
        informationScreen = findViewById(R.id.informationScreen)
        statusText = findViewById(R.id.statusText)
        resultImage = findViewById(R.id.resultImage)
        resultName = findViewById(R.id.resultName)
        resultConfidence = findViewById(R.id.resultConfidence)
        resultMessage = findViewById(R.id.resultMessage)
        captureButton = findViewById(R.id.captureButton)
        continuousButton = findViewById(R.id.continuousButton)
        informationButton = findViewById(R.id.informationButton)
        informationTitle = findViewById(R.id.informationTitle)
        informationSubtitle = findViewById(R.id.informationSubtitle)
        symptomsText = findViewById(R.id.symptomsText)
        preventionText = findViewById(R.id.preventionText)
        helpText = findViewById(R.id.helpText)
        referenceButton = findViewById(R.id.referenceButton)
    }

    private fun bindActions() {
        captureButton.setOnClickListener { capturePhoto() }
        continuousButton.setOnClickListener {
            if (continuousCaptureEnabled) stopContinuousCapture() else startContinuousCapture()
        }
        informationButton.setOnClickListener { latestPrediction?.let { showInformation(it) } }
        findViewById<MaterialButton>(R.id.backButton).setOnClickListener { showCameraScreen() }
        referenceButton.setOnClickListener { openReference() }
    }

    private fun requestCameraWhenNeeded() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            runCatching {
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(cameraPreview.surfaceProvider)
                }
                imageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .setTargetRotation(cameraPreview.display?.rotation ?: Surface.ROTATION_0)
                    .build()
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageCapture
                )
                setStatus(getString(R.string.camera_ready))
            }.onFailure {
                imageCapture = null
                setStatus(getString(R.string.camera_unavailable))
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startContinuousCapture() {
        if (imageCapture == null) {
            setStatus(getString(R.string.camera_unavailable))
            return
        }
        continuousCaptureEnabled = true
        captureButton.isEnabled = false
        continuousButton.text = getString(R.string.stop_capture)
        setStatus(getString(R.string.continuous_status))
        mainHandler.removeCallbacks(continuousRunnable)
        mainHandler.post(continuousRunnable)
    }

    private fun stopContinuousCapture() {
        if (!continuousCaptureEnabled) return
        continuousCaptureEnabled = false
        mainHandler.removeCallbacks(continuousRunnable)
        captureButton.isEnabled = true
        continuousButton.text = getString(R.string.capture_continuous)
        if (!isCaptureOrInferenceRunning) setStatus(getString(R.string.camera_ready))
    }

    private fun capturePhoto() {
        val capture = imageCapture ?: run {
            setStatus(getString(R.string.camera_unavailable))
            return
        }
        if (isCaptureOrInferenceRunning) return

        isCaptureOrInferenceRunning = true
        if (!continuousCaptureEnabled) setStatus(getString(R.string.analysing))
        val outputFile = File(cacheDir, "mango_capture_${System.currentTimeMillis()}.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(outputFile).build()
        capture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    classifyCapturedFile(outputFile)
                }

                override fun onError(exception: ImageCaptureException) {
                    isCaptureOrInferenceRunning = false
                    if (!continuousCaptureEnabled) setStatus(getString(R.string.camera_unavailable))
                }
            }
        )
    }

    private fun handleTestImageIntent(intent: Intent?) {
        intent?.getStringExtra(testImageExtra)
            ?.takeIf { it.isNotBlank() }
            ?.let { classifyCapturedFile(File(it), keepFile = true) }
    }

    private fun classifyCapturedFile(file: File, keepFile: Boolean = false) {
        if (!file.exists()) {
            setStatus(getString(R.string.camera_unavailable))
            return
        }
        isCaptureOrInferenceRunning = true
        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) {
                runCatching { classifyFile(file) }
            }
            isCaptureOrInferenceRunning = false
            result.onSuccess { captured ->
                showPrediction(captured)
                if (!continuousCaptureEnabled) setStatus(getString(R.string.camera_ready))
            }.onFailure {
                setStatus(getString(R.string.camera_unavailable))
            }
            if (!keepFile) file.delete()
        }
    }

    private fun classifyFile(file: File): CapturedPrediction {
        val source = decodeBitmapWithOrientation(file)
        try {
            val prediction = classifyBitmap(source)
            return CapturedPrediction(prediction, createPreviewBitmap(source))
        } finally {
            source.recycle()
        }
    }

    private fun decodeBitmapWithOrientation(file: File): Bitmap {
        val original = BitmapFactory.decodeFile(file.absolutePath)
            ?: error("Không đọc được ảnh")
        val orientation = runCatching {
            ExifInterface(file).getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
        }.getOrDefault(ExifInterface.ORIENTATION_NORMAL)
        val matrix = Matrix().apply {
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> postRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> postRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> postRotate(270f)
            }
        }
        if (matrix.isIdentity) return original
        return Bitmap.createBitmap(original, 0, 0, original.width, original.height, matrix, true)
            .also { original.recycle() }
    }

    private fun createPreviewBitmap(source: Bitmap): Bitmap {
        val maxSide = 512
        val largestSide = maxOf(source.width, source.height)
        val scale = min(1f, maxSide.toFloat() / largestSide.toFloat())
        val width = (source.width * scale).roundToLong().toInt().coerceAtLeast(1)
        val height = (source.height * scale).roundToLong().toInt().coerceAtLeast(1)
        val scaled = Bitmap.createScaledBitmap(source, width, height, true)
        return if (scaled === source) {
            source.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            scaled
        }
    }

    private fun classifyBitmap(bitmap: Bitmap): Prediction {
        val env = OrtEnvironment.getEnvironment()
        val session = getInferenceSession(env)
        val tensorData = bitmapToTensor(bitmap)
        OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(tensorData),
            longArrayOf(1, 3, imageSize.toLong(), imageSize.toLong())
        ).use { tensor ->
            val startedAt = System.nanoTime()
            session.run(mapOf(inputName to tensor)).use { result ->
                val latencyMs = (System.nanoTime() - startedAt) / 1_000_000.0
                val rawOutput = result[0].value
                val logits = when (rawOutput) {
                    is FloatArray -> rawOutput
                    is Array<*> -> rawOutput.firstOrNull() as? FloatArray
                        ?: error("Output model không hợp lệ")
                    else -> error("Output model không hỗ trợ")
                }
                require(logits.size == labels.size) { "Số lớp model không khớp nhãn" }
                val probabilities = softmax(logits)
                val ranking = probabilities.indices.sortedByDescending { probabilities[it] }
                return Prediction(
                    modelLabel = labels[ranking.first()],
                    confidence = probabilities[ranking.first()],
                    latencyMs = latencyMs
                )
            }
        }
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

    private fun loadLabels(): List<String> {
        val json = assets.open(labelsAssetName).bufferedReader().use { it.readText() }
        return Regex("\\\"([^\\\"]+)\\\"").findAll(json)
            .map { it.groupValues[1] }
            .filter { it != "class_names" }
            .toList()
            .also { require(it.size == 7) { "Nhãn model phải có đúng 7 lớp" } }
    }

    private fun bitmapToTensor(bitmap: Bitmap): FloatArray {
        val scale = resizeShortSide.toFloat() / min(bitmap.width, bitmap.height)
        val resizedWidth = (bitmap.width * scale).roundToLong().toInt()
        val resizedHeight = (bitmap.height * scale).roundToLong().toInt()
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

    private fun showPrediction(captured: CapturedPrediction) {
        latestPrediction = captured.prediction
        resultImage.setImageBitmap(captured.previewBitmap)
        resultImage.visibility = View.VISIBLE
        resultName.text = displayName(captured.prediction.modelLabel)
        resultConfidence.text = String.format(Locale.getDefault(), "%.0f%%", captured.prediction.confidence * 100)
        resultConfidence.visibility = View.VISIBLE
        resultMessage.text = confidenceMessage(captured.prediction.confidence)
        informationButton.isEnabled = true
    }

    private fun confidenceMessage(confidence: Float): String = when {
        confidence >= 0.8f -> getString(R.string.confidence_high)
        confidence >= 0.5f -> getString(R.string.confidence_medium)
        else -> getString(R.string.confidence_low)
    }

    private fun showInformation(prediction: Prediction) {
        stopContinuousCapture()
        val content = contentFor(prediction.modelLabel)
        informationTitle.text = getString(R.string.disease_information)
        informationSubtitle.text = displayName(prediction.modelLabel)
        symptomsText.text = content.symptoms
        preventionText.text = content.prevention
        helpText.text = content.seekHelp
        cameraScreen.visibility = View.GONE
        informationScreen.visibility = View.VISIBLE
    }

    private fun showCameraScreen() {
        informationScreen.visibility = View.GONE
        cameraScreen.visibility = View.VISIBLE
    }

    private fun openReference() {
        runCatching {
            startActivity(Intent(Intent.ACTION_VIEW, sourceUrl))
        }.onFailure {
            startActivity(Intent(Settings.ACTION_SETTINGS))
        }
    }

    private fun contentFor(modelLabel: String): DiseaseContent = when (modelLabel) {
        "Anthracnose" -> DiseaseContent(
            "Trên lá có thể xuất hiện đốm nâu hoặc đen, sau đó lan rộng khi điều kiện ẩm kéo dài.",
            "Tỉa cành cho vườn thông thoáng, thu gom lá bệnh và theo dõi vườn sau mưa. Chỉ dùng biện pháp bảo vệ thực vật theo hướng dẫn của cơ quan chuyên môn.",
            "Hỏi cán bộ nông nghiệp địa phương khi vết bệnh lan nhanh, xuất hiện nhiều trên lộc non hoặc ảnh hưởng đến quả."
        )
        "Dieback" -> DiseaseContent(
            "Cành non có thể khô dần từ đầu cành, lá trên cành bị ảnh hưởng héo hoặc rụng.",
            "Cắt tỉa phần cành bị hại bằng dụng cụ sạch, giữ tán cây thông thoáng và theo dõi vết khô mới.",
            "Hỏi cán bộ nông nghiệp khi cành khô lan xuống thân hoặc nhiều cây trong vườn có dấu hiệu tương tự."
        )
        "Gall Midge" -> DiseaseContent(
            "Lá non có thể biến dạng, có vết hại nhỏ hoặc phát triển không đều.",
            "Theo dõi đợt lộc non, vệ sinh vườn và trao đổi với cán bộ kỹ thuật về biện pháp phù hợp tại địa phương.",
            "Hỏi cán bộ nông nghiệp khi lá non bị hại trên diện rộng hoặc cây giảm sinh trưởng rõ rệt."
        )
        "Healthy" -> DiseaseContent(
            "Ảnh hiện tại chưa cho thấy dấu hiệu rõ của sáu nhóm bệnh còn lại trong model 7 lớp.",
            "Tiếp tục giữ vườn thông thoáng, theo dõi lá non và chụp lại khi xuất hiện đốm, mốc hoặc biến dạng.",
            "Hỏi cán bộ nông nghiệp nếu cây vẫn suy yếu, dù app không nhận diện được dấu hiệu trong ảnh."
        )
        "Powdery Mildew" -> DiseaseContent(
            "Bề mặt lá có thể có lớp phấn trắng hoặc mốc mịn, thường dễ thấy hơn trên phần non.",
            "Giữ tán cây thông thoáng, tránh để ẩm kéo dài và theo dõi sớm các đợt lá non.",
            "Hỏi cán bộ nông nghiệp khi lớp phấn lan nhanh hoặc xuất hiện đồng thời trên hoa, chồi và lá."
        )
        "Red Rust" -> DiseaseContent(
            "Lá có thể xuất hiện mảng màu cam đỏ hoặc nâu đỏ, cần chụp thêm mặt trên và mặt dưới lá để đối chiếu.",
            "Thu gom bộ phận bị hại, giữ vườn sạch và theo dõi diễn biến sau các giai đoạn ẩm kéo dài.",
            "Hỏi cán bộ nông nghiệp khi mảng màu lan rộng, lá rụng nhiều hoặc cây sinh trưởng kém."
        )
        else -> DiseaseContent(
            "Lá có thể phủ lớp đen như muội; nên kiểm tra thêm sự xuất hiện của côn trùng tiết mật trên cây.",
            "Giữ tán cây thông thoáng, vệ sinh lá bị ảnh hưởng và hỏi cán bộ kỹ thuật về cách quản lý côn trùng liên quan.",
            "Hỏi cán bộ nông nghiệp khi lớp đen phủ rộng hoặc xuất hiện đồng thời rệp, rầy hay côn trùng chích hút."
        )
    }

    private fun displayName(modelLabel: String): String = when (modelLabel) {
        "Anthracnose" -> getString(R.string.anthracnose_name)
        "Dieback" -> getString(R.string.dieback_name)
        "Gall Midge" -> getString(R.string.gall_midge_name)
        "Healthy" -> getString(R.string.healthy_name)
        "Powdery Mildew" -> getString(R.string.powdery_mildew_name)
        "Red Rust" -> getString(R.string.red_rust_name)
        else -> getString(R.string.sooty_mold_name)
    }

    private fun setStatus(message: String) {
        statusText.text = message
    }

    private data class Prediction(
        val modelLabel: String,
        val confidence: Float,
        val latencyMs: Double
    )

    private data class CapturedPrediction(
        val prediction: Prediction,
        val previewBitmap: Bitmap
    )

    private data class DiseaseContent(
        val symptoms: String,
        val prevention: String,
        val seekHelp: String
    )
}
