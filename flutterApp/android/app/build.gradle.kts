plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "io.github.miuzarte.inferencer_flutter"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "io.github.miuzarte.inferencer_flutter"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    // QNN HTP：Skel 库必须以真实文件落盘（fastrpc 走文件路径）
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

flutter {
    source = "../.."
}

dependencies {
    // 官方插件把 QNN 运行时声明为 compileOnly，宿主 App 必须显式引入：
    // ORT 1.26（SM8850 支持）+ QAIRT 2.46（设备表含 SM8850）
    implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.26.0")
    implementation("com.qualcomm.qti:qnn-runtime:2.46.0")
}
