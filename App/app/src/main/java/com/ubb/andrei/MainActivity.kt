package com.ubb.andrei

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.io.File
import okhttp3.*
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter


private const val FILE_NAME = "photo.jpg"
private lateinit var photoFile: File

class MainActivity : AppCompatActivity() {

    private val REQUEST_PERMISSION = 10
    //private val REQUEST_IMAGE_CAPTURE = 1
    //private val REQUEST_PICK_IMAGE = 2

    var btnTakePicture: Button? = null
    var btnOpenGallery: Button? = null
    var btnUploadPicture: Button? = null
    var imageView: ImageView? = null

    var stringPhoto : String? = null
    var uriPhoto: Uri? = null
    var isTaken: Boolean? = null

    private fun checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.CAMERA),
                REQUEST_PERMISSION)
        }
    }

    private fun rotateImage(source: Bitmap, angle: Int): Bitmap? {
        val matrix = Matrix()
        matrix.postRotate(angle.toFloat())
        return Bitmap.createBitmap(
            source, 0, 0, source.width, source.height,
            matrix, true
        )
    }

    private fun getRotatedBitmap(filePath: String): Bitmap? {
        val takenImage = BitmapFactory.decodeFile(filePath)

        val ei = ExifInterface(filePath)

        val orientation: Int = ei.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_UNDEFINED
        )

        var rotatedBitmap: Bitmap? = null
        rotatedBitmap = when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> rotateImage(takenImage, 90)
            ExifInterface.ORIENTATION_ROTATE_180 -> rotateImage(takenImage, 180)
            ExifInterface.ORIENTATION_ROTATE_270 -> rotateImage(takenImage, 270)
            ExifInterface.ORIENTATION_NORMAL -> takenImage
            else -> takenImage
        }

        return rotatedBitmap
    }


    var resultLauncherTakePhoto = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            //val takenImage = result.data?.extras?.get("data") as Bitmap
            //val takenImage = BitmapFactory.decodeFile(photoFile.absolutePath)
            val rotatedBitmap = getRotatedBitmap(photoFile.absolutePath)

            imageView?.setImageBitmap(rotatedBitmap)
            isTaken = true
            stringPhoto = photoFile.absolutePath
            btnUploadPicture?.isEnabled = true
        }
    }

    var resultLauncherOpenGallery = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if(result.resultCode == Activity.RESULT_OK) {
            val chosenImage = result?.data?.getData()

            val pathFromUri = chosenImage?.let { URIPathHelper().getPath(this, it) }
            val takenImage = pathFromUri?.let { getRotatedBitmap(it) }

            //imageView?.setImageURI(chosenImage)
            imageView?.setImageBitmap(takenImage)
            isTaken = false
            uriPhoto = chosenImage
            btnUploadPicture?.isEnabled = true
        }
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnTakePicture = findViewById<Button>(R.id.btnTakePicture)
        btnOpenGallery = findViewById<Button>(R.id.btnOpenGallery)
        btnUploadPicture = findViewById<Button>(R.id.btnUploadPicture)
        imageView = findViewById<ImageView>(R.id.imageView)

        btnTakePicture?.setOnClickListener {
            buttonTakePictureClicked()
        }

        btnOpenGallery?.setOnClickListener {
            buttonOpenGalleryClicked()
        }

        btnUploadPicture?.setOnClickListener {

            buttonUploadPictureClicked()
        }
    }

    fun testConnection(){
        Thread {
            val client = OkHttpClient()
            val serverURL: String = "http://192.168.0.148:420"
            try {
                val formBody = FormBody.Builder()
                    .add("username", "test")
                    .add("password", "test")
                    .build()

                val request: Request = Request.Builder()
                    .url("$serverURL/uploadPhoto")
                    .post(formBody)
                    .build()

                val call: Call = client.newCall(request)

                val response: Response = call.execute()

                if (response.isSuccessful) {
                    Log.d("File upload", "success")
                    Toast.makeText(this, "success", Toast.LENGTH_LONG).show()
                } else {
                    Log.e("File upload", "failed")
                    Toast.makeText(this, "failed", Toast.LENGTH_LONG).show()
                }
            } catch (ex: Exception) {
                ex.printStackTrace()
                Log.e("File upload", "failed")
                Toast.makeText(this, "failed + ${ex.message}", Toast.LENGTH_LONG).show()
            }
        }.start()
    }

    fun buttonTakePictureClicked(){
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        photoFile = getPhotoFile(FILE_NAME)


        val fileProvider = FileProvider.getUriForFile(this, "com.ubb.andrei.fileprovider", photoFile)
        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, fileProvider)
        resultLauncherTakePhoto.launch(takePictureIntent)
        //startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)

    }

    fun buttonOpenGalleryClicked(){
        val openGalleryIntent = Intent(Intent.ACTION_PICK)
        openGalleryIntent.type = "image/*"

        resultLauncherOpenGallery.launch(openGalleryIntent)
        //startActivityForResult(openGalleryIntent, REQUEST_PICK_IMAGE)
    }

    fun buttonUploadPictureClicked(){
        val currentTime = LocalDateTime.now()
        val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss-SSS")
        val formatted = currentTime.format(formatter)
        val photoName = "$formatted.jpg"

        if(isTaken == true) {
            UploadUtility(this).uploadFile(stringPhoto!!, photoName)
            //UploadUtility(this).uploadFile(stringPhoto!!)
        }
        if(isTaken == false){
            UploadUtility(this).uploadFile(uriPhoto!!, photoName)
            //UploadUtility(this).uploadFile(uriPhoto!!)
        }
    }

    /*
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (resultCode == RESULT_OK) {
            if (requestCode == REQUEST_IMAGE_CAPTURE) {
                //val uri = Uri.parse(currentPhotoPath)
                //ivImage.setImageURI(uri)

                val takenImage = data?.extras?.get("data") as Bitmap
                imageView?.setImageBitmap(takenImage)
            }
            else if (requestCode == REQUEST_PICK_IMAGE) {
                val imageUri = data?.getData()
                imageView?.setImageURI(imageUri)
            }
        }

        super.onActivityResult(requestCode, resultCode, data)
    }
    */

    override fun onResume() {
        super.onResume()
        checkCameraPermission()
    }

    private fun getPhotoFile(fileName: String): File {
        //Use 'getExternalFilesDir' on Context to access package-specific directories
        val storageDirectory = getExternalFilesDir(Environment.DIRECTORY_PICTURES)

        return File.createTempFile(fileName, ".jpg", storageDirectory)
    }

}