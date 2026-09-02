plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.mangomamba.benchmark"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.mangomamba.benchmark"
        minSdk = 26
        targetSdk = 34
        versionCode = 2
        versionName = "0.2-mixed"
    }

    // Model is fp32 and >20MB; don't let AGP compress it away from being
    // readable as a raw asset stream.
    androidResources {
        noCompress += "onnx"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }

    packaging {
        resources.excludes += "META-INF/**"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // ONNX Runtime for Android. Ships CPU EP + XNNPACK.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")

    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    implementation("androidx.camera:camera-core:1.3.4")
    implementation("androidx.camera:camera-camera2:1.3.4")
    implementation("androidx.camera:camera-lifecycle:1.3.4")
    implementation("androidx.camera:camera-view:1.3.4")
    implementation("androidx.exifinterface:exifinterface:1.3.7")
}
