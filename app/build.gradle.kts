plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "io.github.aoihoshino.realcugan_ncnn_android"
    compileSdk = 35

    defaultConfig {
        applicationId = "io.github.aoihoshino.realcugan_ncnn_android"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // 保留完整的 .so 符号，方便 ndk-stack 符号化 tombstone
        ndk {
            debugSymbolLevel = "FULL"
        }

        externalNativeBuild {
            cmake {
                // 传给 CMake 的参数列表
                arguments += listOf(
                    "-DANDROID_TOOLCHAIN=clang",
                    "-DANDROID_STL=c++_static",
                    "-DCMAKE_BUILD_TYPE=Debug"    // 确保是 Debug 模式
                )
                // 额外的编译标志（-g 开启调试符号）
                cppFlags += listOf("-g")
            }
        }
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
            isJniDebuggable = true
            externalNativeBuild {
                cmake {
                    // Debug 构建再确认一次
                    arguments += "-DCMAKE_BUILD_TYPE=Debug"
                }
            }
        }
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation(project(":realcugan-android-sdk"))
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}