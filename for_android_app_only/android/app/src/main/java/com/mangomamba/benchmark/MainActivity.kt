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
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton
import com.google.android.material.progressindicator.LinearProgressIndicator
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
    private val testBatchManifestExtra = "test_batch_manifest_path"
    private val captureIntervalMs = 800L
    private lateinit var cameraPreview: PreviewView
    private lateinit var cameraScreen: LinearLayout
    private lateinit var informationScreen: LinearLayout
    private lateinit var statusText: TextView
    private lateinit var inferenceProgress: LinearProgressIndicator
    private lateinit var resultImage: ImageView
    private lateinit var resultPlaceholderIcon: ImageView
    private lateinit var resultName: TextView
    private lateinit var resultConfidence: TextView
    private lateinit var resultMessage: TextView
    private lateinit var captureButton: MaterialButton
    private lateinit var pickImageButton: MaterialButton
    private lateinit var continuousButton: MaterialButton
    private lateinit var informationButton: MaterialButton
    private lateinit var informationSubtitle: TextView
    private lateinit var informationEnglishName: TextView
    private lateinit var diseaseGuideImage: ImageView
    private lateinit var symptomsText: TextView
    private lateinit var preventionText: TextView
    private lateinit var referenceButton: MaterialButton
    private lateinit var referenceButtonSecondary: MaterialButton

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

    private val galleryPickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let(::classifyGalleryImage)
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
        inferenceProgress = findViewById(R.id.inferenceProgress)
        resultImage = findViewById(R.id.resultImage)
        resultPlaceholderIcon = findViewById(R.id.resultPlaceholderIcon)
        resultName = findViewById(R.id.resultName)
        resultConfidence = findViewById(R.id.resultConfidence)
        resultMessage = findViewById(R.id.resultMessage)
        captureButton = findViewById(R.id.captureButton)
        pickImageButton = findViewById(R.id.pickImageButton)
        continuousButton = findViewById(R.id.continuousButton)
        informationButton = findViewById(R.id.informationButton)
        informationSubtitle = findViewById(R.id.informationSubtitle)
        informationEnglishName = findViewById(R.id.informationEnglishName)
        diseaseGuideImage = findViewById(R.id.diseaseGuideImage)
        symptomsText = findViewById(R.id.symptomsText)
        preventionText = findViewById(R.id.preventionText)
        referenceButton = findViewById(R.id.referenceButton)
        referenceButtonSecondary = findViewById(R.id.referenceButtonSecondary)
    }

    private fun bindActions() {
        captureButton.setOnClickListener { capturePhoto() }
        pickImageButton.setOnClickListener { galleryPickerLauncher.launch("image/*") }
        continuousButton.setOnClickListener {
            if (continuousCaptureEnabled) stopContinuousCapture() else startContinuousCapture()
        }
        informationButton.setOnClickListener { latestPrediction?.let { showInformation(it) } }
        findViewById<MaterialToolbar>(R.id.informationTopAppBar)
            .setNavigationOnClickListener { showCameraScreen() }
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
        showInferenceProgress(true)
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
                    showInferenceProgress(false)
                    if (!continuousCaptureEnabled) setStatus(getString(R.string.camera_unavailable))
                }
            }
        )
    }

    private fun classifyGalleryImage(uri: Uri) {
        if (isCaptureOrInferenceRunning) return
        val imageFile = File(cacheDir, "mango_album_${System.currentTimeMillis()}.jpg")
        runCatching {
            contentResolver.openInputStream(uri)?.use { input ->
                imageFile.outputStream().use { output -> input.copyTo(output) }
            } ?: error("Không đọc được ảnh đã chọn")
        }.onSuccess {
            classifyCapturedFile(imageFile)
        }.onFailure {
            imageFile.delete()
            setStatus(getString(R.string.image_unavailable))
        }
    }

    private fun handleTestImageIntent(intent: Intent?) {
        intent?.getStringExtra(testBatchManifestExtra)
            ?.takeIf { it.isNotBlank() }
            ?.let { runBatchInference(File(it)) }
            ?: intent?.getStringExtra(testImageExtra)
            ?.takeIf { it.isNotBlank() }
            ?.let { classifyCapturedFile(File(it), keepFile = true) }
    }

    private fun runBatchInference(manifestFile: File) {
        if (isCaptureOrInferenceRunning || !manifestFile.isFile) return
        val entries = manifestFile.readLines()
            .mapNotNull { line ->
                val fields = line.split('\t', limit = 2)
                val expectedLabel = fields.firstOrNull()?.removePrefix("\uFEFF")
                if (fields.size == 2 && expectedLabel != null && expectedLabel in labels && File(fields[1]).isFile) {
                    BatchEntry(expectedLabel, File(fields[1]))
                } else {
                    null
                }
            }
        if (entries.isEmpty()) return

        isCaptureOrInferenceRunning = true
        showInferenceProgress(true)
        setStatus("Đang kiểm tra 0/${entries.size} ảnh")
        lifecycleScope.launch {
            val summary = withContext(Dispatchers.Default) {
                val outputFile = File(getExternalFilesDir(null), "vn_mangoleaf_batch_results.csv")
                val correctByClass = mutableMapOf<String, Int>()
                val countByClass = mutableMapOf<String, Int>()
                outputFile.bufferedWriter().use { writer ->
                    writer.appendLine("expected,predicted,confidence,latency_ms,file_name")
                    entries.forEachIndexed { index, entry ->
                        val captured = classifyFile(entry.file)
                        try {
                            val prediction = captured.prediction
                            countByClass[entry.expectedLabel] = (countByClass[entry.expectedLabel] ?: 0) + 1
                            if (prediction.modelLabel == entry.expectedLabel) {
                                correctByClass[entry.expectedLabel] = (correctByClass[entry.expectedLabel] ?: 0) + 1
                            }
                            writer.appendLine(
                                listOf(
                                    entry.expectedLabel,
                                    prediction.modelLabel,
                                    String.format(Locale.US, "%.6f", prediction.confidence),
                                    String.format(Locale.US, "%.2f", prediction.latencyMs),
                                    entry.file.name
                                ).joinToString(",")
                            )
                        } finally {
                            captured.previewBitmap.recycle()
                        }
                        if ((index + 1) % 25 == 0 && index + 1 < entries.size) {
                            runOnUiThread { setStatus("Đang kiểm tra ${index + 1}/${entries.size} ảnh") }
                        }
                    }
                }
                BatchSummary(outputFile, countByClass, correctByClass)
            }
            isCaptureOrInferenceRunning = false
            showInferenceProgress(false)
            val correct = summary.correctByClass.values.sum()
            setStatus("Đã kiểm tra ${entries.size} ảnh, đúng $correct/${entries.size}. CSV đã lưu trong thư mục app.")
        }
    }

    private fun classifyCapturedFile(file: File, keepFile: Boolean = false) {
        if (!file.exists()) {
            setStatus(getString(R.string.camera_unavailable))
            return
        }
        isCaptureOrInferenceRunning = true
        showInferenceProgress(true)
        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) {
                runCatching { classifyFile(file) }
            }
            isCaptureOrInferenceRunning = false
            showInferenceProgress(false)
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
        resultPlaceholderIcon.visibility = View.GONE
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
        val content = factCheckedContentFor(prediction.modelLabel)
        informationSubtitle.text = displayName(prediction.modelLabel)
        informationEnglishName.text = content.englishName
        diseaseGuideImage.setImageResource(content.guideImageRes)
        symptomsText.text = content.symptoms
        preventionText.text = content.prevention
        bindReferenceButton(referenceButton, content.references.first())
        content.references.getOrNull(1)?.let { secondReference ->
            bindReferenceButton(referenceButtonSecondary, secondReference)
            referenceButtonSecondary.visibility = View.VISIBLE
        } ?: run {
            referenceButtonSecondary.visibility = View.GONE
        }
        cameraScreen.visibility = View.GONE
        informationScreen.visibility = View.VISIBLE
    }

    private fun showCameraScreen() {
        informationScreen.visibility = View.GONE
        cameraScreen.visibility = View.VISIBLE
    }

    private fun bindReferenceButton(button: MaterialButton, reference: ReferenceLink) {
        button.text = reference.title
        button.setOnClickListener { openReference(reference.uri) }
    }

    private fun openReference(uri: Uri) {
        runCatching {
            startActivity(Intent(Intent.ACTION_VIEW, uri))
        }.onFailure {
            referenceButton.error = getString(R.string.camera_unavailable)
        }
    }

    private fun factCheckedContentFor(modelLabel: String): FactCheckedDiseaseContent = when (modelLabel) {
        "Anthracnose" -> FactCheckedDiseaseContent(
            englishName = "Anthracnose",
            guideImageRes = R.drawable.guide_anthracnose,
            symptoms = "• Đốm nâu đen thường xuất hiện trước trên lá non; vùng lá lành quanh vết có thể vàng.\n• Vết bệnh lớn dần, liên kết với nhau và làm phần lá bị cháy hoặc khô.\n• Cành non có thể có vết nâu đen hơi lõm và khô dần.\n• Bệnh dễ phát triển khi vườn ẩm, nhiều sương hoặc mưa kéo dài.",
            prevention = "• Thu gom và tiêu hủy lá, cành bệnh; dọn cỏ và tàn dư trong vườn.\n• Cắt tỉa cành vô hiệu để tán nhận ánh sáng và nhanh khô sau mưa.\n• Bón phân cân đối, tránh bón thừa đạm làm tán quá rậm.\n• Theo dõi sớm ở đợt lộc non, nụ và quả non; chỉ dùng thuốc nằm trong danh mục và theo nhãn.",
            references = listOf(
                ReferenceLink("Sở NN&MT Lai Châu", "https://sonnmt.laichau.gov.vn/tin-tuc-thong-ke/tang-cuong-phong-tru-sau-benh-hai-tren-cay-xoai.html"),
                ReferenceLink("Viện Cây ăn quả miền Nam", "https://sofri.org.vn/vi/quy-trinh-ky-thuat-tham-canh-tong-hop-cay-xoai-cho-cac-tinh-nam-bo.html")
            )
        )
        "Dieback" -> FactCheckedDiseaseContent(
            englishName = "Dieback",
            guideImageRes = R.drawable.guide_dieback,
            symptoms = "• Đọt hoặc cành non có vết sậm màu, sau đó khô dần từ phần ngọn.\n• Lá trên cành bị hại chuyển nâu, có thể cuốn mép và rụng sớm.\n• Vết hại có thể lan xuống cành; cành nặng bị nhăn hoặc chết khô.\n• Cần phân biệt với khô cành do thiếu nước hay tổn thương cơ giới bằng cách theo dõi vết mới.",
            prevention = "• Dùng giống, cành ghép khỏe và vệ sinh dao kéo trước, sau khi cắt tỉa.\n• Cắt bỏ phần cành bệnh vào lúc khô ráo, thu gom và tiêu hủy ngoài vườn.\n• Giữ đất thoát nước, tránh để vùng rễ ngập úng kéo dài.\n• Bón phân cân đối và theo dõi sớm các đọt mới để xử lý ổ bệnh nhỏ.",
            references = listOf(
                ReferenceLink("Nông nghiệp AVN", "https://nongnghiepavn.com/phong-tru-benh-kho-dot-thoi-trai-tren-cay-xoai/"),
                ReferenceLink("Cổng TTĐT Tiền Giang", "https://tiengiang.gov.vn/hoi-dap/-/qa/3?_questionmvcportletcontentdgaction_WAR_Iportal_Hoi_Dap_Manageportlet_cur=30")
            )
        )
        "Gall Midge" -> FactCheckedDiseaseContent(
            englishName = "Gall Midge",
            guideImageRes = R.drawable.guide_gall_midge,
            symptoms = "• Trên lá có thể thấy nốt u hoặc mụn, phần mô lá sưng phồng.\n• Lá non bị hại nặng dễ vàng, biến dạng hoặc phát triển không đều.\n• Cần kiểm tra kỹ đợt lá non vì đây là giai đoạn dễ nhìn thấy dấu hiệu nhất.\n• Tổn thương tương tự có thể xuất hiện ở chồi, hoa hoặc quả non.",
            prevention = "• Cắt tỉa tạo tán thông thoáng để dễ quan sát đợt lộc non.\n• Điều khiển cây ra đọt tập trung, tránh nhiều đợt lá non chồng chéo khó quản lý.\n• Theo dõi thường xuyên ngay khi lá non mới mở; loại bỏ bộ phận hại nặng.\n• Khi cần dùng biện pháp bảo vệ thực vật, chọn sản phẩm được phép cho cây xoài và theo đúng nhãn.",
            references = listOf(
                ReferenceLink("Viện Cây ăn quả miền Nam", "https://sofri.org.vn/vi/tin-tuc/quy-trinh-khoa-hoc-cong-nghe/quy-trinh-ky-thuat-tham-canh-tong-hop-cay-xoai-cho-cac-tinh-nam-bo-745.html?download=2&id=0"),
                ReferenceLink("Cẩm Nang Nhà Nông", "https://camnangnhanong.com/trong-trot/cay-an-qua/la-xoai-bi-noi-mun-la-bi-benh-gi.html")
            )
        )
        "Healthy" -> FactCheckedDiseaseContent(
            englishName = "Healthy leaf",
            guideImageRes = R.drawable.guide_healthy,
            symptoms = "• Ảnh hiện tại không có dấu hiệu nổi bật của sáu nhóm bệnh còn lại trong model.\n• Lá khỏe thường có màu tương đối đồng đều, không có lớp phấn trắng hay mảng muội đen.\n• Không nên coi kết quả này là chẩn đoán toàn bộ cây vì ảnh chỉ phản ánh phần lá đã chụp.\n• Chụp lại khi lá xuất hiện đốm, mốc, mụn hoặc biến dạng mới.",
            prevention = "• Giữ tán cây thông thoáng và dọn lá, cành già sau thu hoạch.\n• Theo dõi lá non, mặt dưới lá và các côn trùng chích hút định kỳ.\n• Bón phân, tưới và thoát nước phù hợp với điều kiện vườn.\n• Tiếp tục ghi nhận ảnh ở cùng vị trí để phát hiện thay đổi sớm.",
            references = listOf(ReferenceLink("Viện Cây ăn quả miền Nam", "https://sofri.org.vn/vi/quy-trinh-ky-thuat-tham-canh-tong-hop-cay-xoai-cho-cac-tinh-nam-bo.html"))
        )
        "Powdery Mildew" -> FactCheckedDiseaseContent(
            englishName = "Powdery Mildew",
            guideImageRes = R.drawable.guide_powdery_mildew,
            symptoms = "• Mặt lá, hoa hoặc chùm quả non có thể phủ lớp phấn trắng như bột.\n• Phần mô non bị hại suy yếu, hoa dễ khô và giảm khả năng đậu quả.\n• Dấu hiệu thường rõ trong giai đoạn cây ra hoa, đậu quả.\n• Khi phấn lan rộng, kiểm tra cả chồi non và chùm hoa để đánh giá mức độ.",
            prevention = "• Thu gom bộ phận rụng, cắt bỏ cành già và phần nhiễm bệnh sau thu hoạch.\n• Tạo tán thoáng, giảm nơi giữ ẩm và giúp kiểm tra phát hoa dễ hơn.\n• Bón phân cân đối, không thúc đạm quá mức.\n• Theo dõi từ lúc phát hoa; nếu cần xử lý thuốc, dùng theo nhãn và tránh ảnh hưởng thụ phấn.",
            references = listOf(
                ReferenceLink("Sở NN&MT Lai Châu", "https://sonnmt.laichau.gov.vn/tin-tuc-thong-ke/tang-cuong-phong-tru-sau-benh-hai-tren-cay-xoai.html"),
                ReferenceLink("Viện Cây ăn quả miền Nam", "https://sofri.org.vn/vi/tin-tuc/quy-trinh-khoa-hoc-cong-nghe/quy-trinh-ky-thuat-tham-canh-tong-hop-cay-xoai-cho-cac-tinh-nam-bo-745.html?download=2&id=0")
            )
        )
        "Red Rust" -> FactCheckedDiseaseContent(
            englishName = "Red Rust / Algal Leaf Spot",
            guideImageRes = R.drawable.guide_red_rust,
            symptoms = "• Trên lá có đốm tròn cam, nâu đỏ hoặc màu rỉ sắt.\n• Đốm thường hơi nhô lên, bề mặt có cảm giác như lớp nhung và dễ thấy ở mặt trên lá.\n• Các đốm có thể liên kết thành mảng khi bệnh nặng.\n• Mảng bệnh rộng làm giảm diện tích quang hợp của lá.",
            prevention = "• Trồng với mật độ hợp lý, cắt tỉa để ánh sáng và gió đi qua tán.\n• Thoát nước tốt trong mùa mưa, không để vườn ẩm kéo dài.\n• Tỉa và tiêu hủy lá, cành bị nặng để giảm nguồn bệnh.\n• Duy trì chăm sóc và dinh dưỡng cân đối để cây phục hồi tốt.",
            references = listOf(
                ReferenceLink("Khuyến nông Việt Nam", "https://khuyennongvn.gov.vn/khoa-hoc-cong-nghe/khcn-trong-nuoc/phong-tri-benh-dom-rong-tren-cay-an-trai-trong-mua-mua-17309.html"),
                ReferenceLink("Chi cục TT&BVTV TP.HCM", "https://chicucttbvtvhcm.gov.vn/cay-an-qua/benh-dom-rong-471.html")
            )
        )
        else -> FactCheckedDiseaseContent(
            englishName = "Sooty Mold",
            guideImageRes = R.drawable.guide_sooty_mold,
            symptoms = "• Bề mặt lá có lớp muội đen hoặc mảng đen như bồ hóng.\n• Lớp nấm thường phát triển trên mật ngọt do rầy, rệp hay côn trùng chích hút tiết ra.\n• Lá bị phủ dày nhận ánh sáng kém và giảm quang hợp.\n• Nên kiểm tra mặt dưới lá, chồi và cuống lá để tìm rầy, rệp đi kèm.",
            prevention = "• Ưu tiên quản lý rầy, rệp và các côn trùng chích hút tạo mật ngọt.\n• Cắt tỉa cành để tán thông thoáng và thuận tiện kiểm tra ổ dịch.\n• Theo dõi lá non, chồi non và cuống lá, nơi côn trùng thường tập trung.\n• Chăm sóc cây khỏe, chỉ chọn biện pháp bảo vệ thực vật theo danh mục và hướng dẫn trên nhãn.",
            references = listOf(
                ReferenceLink("Chi cục TT&BVTV TP.HCM", "https://chicucttbvtvhcm.gov.vn/cay-an-qua/benh-bo-hong-493.html"),
                ReferenceLink("Viện Cây ăn quả miền Nam", "https://sofri.org.vn/vi/tin-tuc/quy-trinh-khoa-hoc-cong-nghe/quy-trinh-ky-thuat-tham-canh-tong-hop-cay-xoai-cho-cac-tinh-nam-bo-745.html?download=2&id=0")
            )
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

    private fun showInferenceProgress(show: Boolean) {
        inferenceProgress.visibility = if (show) View.VISIBLE else View.GONE
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

    private data class BatchEntry(val expectedLabel: String, val file: File)

    private data class BatchSummary(
        val outputFile: File,
        val countByClass: Map<String, Int>,
        val correctByClass: Map<String, Int>
    )

    private data class FactCheckedDiseaseContent(
        val englishName: String,
        val guideImageRes: Int,
        val symptoms: String,
        val prevention: String,
        val references: List<ReferenceLink>
    )

    private data class ReferenceLink(val title: String, val url: String) {
        val uri: Uri get() = Uri.parse(url)
    }
}
