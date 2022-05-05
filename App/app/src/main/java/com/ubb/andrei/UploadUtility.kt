package com.ubb.andrei

import android.app.Activity
import android.app.ProgressDialog
import android.net.Uri
import android.util.Log
import android.webkit.MimeTypeMap
import android.widget.Toast
import com.google.gson.Gson
import com.ubb.andrei.domain.ServerResponse
import com.ubb.andrei.utils.IObservable
import com.ubb.andrei.utils.IObserver
import com.ubb.andrei.utils.URIPathHelper
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File

class UploadUtility(activity: Activity, override val observers: ArrayList<IObserver>): IObservable {

    var activity = activity;
    var dialog: ProgressDialog? = null
    var serverURL: String = "http://192.168.0.148:420"
    // var serverURL: String = "http://10.0.2.2:5000"
    var serverUploadDirectoryPath: String = "E:\\Facultate\\Licenta\\bachelors-project-ubb\\ServerPhotos"
    val client = OkHttpClient()
    lateinit var guess : ServerResponse

    fun uploadFile(sourceFilePath: String, uploadedFileName: String? = null) {
        return uploadFile(File(sourceFilePath), uploadedFileName)
    }

    fun uploadFile(sourceFileUri: Uri, uploadedFileName: String? = null) {
        val pathFromUri = URIPathHelper().getPath(activity, sourceFileUri)
        return uploadFile(File(pathFromUri), uploadedFileName)
    }

    fun uploadFile(sourceFile: File, uploadedFileName: String? = null) {
        Thread {
            val mimeType = getMimeType(sourceFile);
            if (mimeType == null) {
                Log.e("file error", "Not able to get mime type")
                guess = ServerResponse(-1, "Error", 0.0, "Fail", "")
                return@Thread
            }
            val fileName: String = uploadedFileName ?: sourceFile.name
            toggleProgressDialog(true)
            try {
                val requestBody: RequestBody =
                    MultipartBody.Builder().setType(MultipartBody.FORM)
                        .addFormDataPart(
                            "uploaded_file",
                            fileName,
                            sourceFile.asRequestBody(mimeType.toMediaTypeOrNull())
                        )
                        .build()

                val request: Request = Request.Builder().url("$serverURL/uploadPhoto")
                    .post(requestBody).build()

                val response: Response = client.newCall(request).execute()

                Log.d("A", response.toString())
                if (response.body!=null){
                    val jsonString = response.body!!.string()
                    val gson = Gson()
                    guess = gson.fromJson(jsonString, ServerResponse::class.java)
                }

                if (response.isSuccessful) {
                    Log.d("Y", guess.toString())
                    sendUpdateEvent(guess)

                    Log.d("File upload", "success, path: $serverUploadDirectoryPath$fileName, response is: $guess.name")
                    showToast("File uploaded successfully, response is: $guess.name")
                } else {
                    Log.e("File upload", "failed")
                    showToast("File uploading failed")
                }

            } catch (ex: Exception) {
                ex.printStackTrace()
                Log.e("File upload", "failed")
                showToast("File uploading failed")
                guess = ServerResponse(-1, "Error", 0.0, "Fail", "")
            }
            toggleProgressDialog(false)
        }.start()
    }

    fun uploadRecyclingInfo(correctPlastic: String, filename: String) {
        Log.d("U", "$correctPlastic - $filename")
        Thread {
            try {
                val requestBody: RequestBody = FormBody.Builder()
                    .add("plastic", correctPlastic)
                    .add("filename", filename)
                    .build()

                val request: Request = Request.Builder().url("$serverURL/uploadInfo")
                    .post(requestBody).build()

                val response: Response = client.newCall(request).execute()

                if (response.isSuccessful) {
                    Log.d("Info upload", "success: $correctPlastic, $filename")
                } else {
                    Log.e("Info upload", "failed")
                }
            } catch (ex: Exception) {
                ex.printStackTrace()
                Log.e("File upload", "failed")
            }
        }.start()
    }

    // url = file path or whatever suitable URL you want.
    fun getMimeType(file: File): String? {
        var type: String? = null
        val extension = MimeTypeMap.getFileExtensionFromUrl(file.path)
        if (extension != null) {
            type = MimeTypeMap.getSingleton().getMimeTypeFromExtension(extension)
        }
        return type
    }

    fun showToast(message: String) {
        activity.runOnUiThread {
            Toast.makeText(activity, message, Toast.LENGTH_LONG).show()
        }
    }

    fun toggleProgressDialog(show: Boolean) {
        activity.runOnUiThread {
            if (show) {
                dialog = ProgressDialog.show(activity, "", "Uploading file...", true);
            } else {
                dialog?.dismiss();
            }
        }
    }
}