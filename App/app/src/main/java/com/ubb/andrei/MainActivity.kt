package com.ubb.andrei

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.io.File


private const val FILE_NAME = "photo.jpg"
private lateinit var photoFile: File

class MainActivity : AppCompatActivity() {

    private val REQUEST_PERMISSION = 10
    private val REQUEST_IMAGE_CAPTURE = 1
    private val REQUEST_PICK_IMAGE = 2

    var btnTakePicture: Button? = null
    var btnOpenGallery: Button? = null
    var imageView: ImageView? = null

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


    var resultLauncherTakePhoto = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            //val takenImage = result.data?.extras?.get("data") as Bitmap
            val takenImage = BitmapFactory.decodeFile(photoFile.absolutePath)

            val ei = ExifInterface(photoFile.absolutePath)
            val orientation: Int = ei.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_UNDEFINED
            )

            var rotatedBitmap: Bitmap? = null
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> rotatedBitmap = rotateImage(takenImage, 90)
                ExifInterface.ORIENTATION_ROTATE_180 -> rotatedBitmap = rotateImage(takenImage, 180)
                ExifInterface.ORIENTATION_ROTATE_270 -> rotatedBitmap = rotateImage(takenImage, 270)
                ExifInterface.ORIENTATION_NORMAL -> rotatedBitmap = takenImage
                else -> rotatedBitmap = takenImage
            }

            imageView?.setImageBitmap(rotatedBitmap)
        }
    }

    var resultLauncherOpenGallery = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if(result.resultCode == Activity.RESULT_OK) {
            val chosenImage = result?.data?.getData()

            imageView?.setImageURI(chosenImage)
        }
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnTakePicture = findViewById<Button>(R.id.btnTakePicture)
        btnOpenGallery = findViewById<Button>(R.id.btnOpenGallery)
        imageView = findViewById<ImageView>(R.id.imageView)

        btnTakePicture?.setOnClickListener {
            val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            photoFile = getPhotoFile(FILE_NAME)


            val fileProvider = FileProvider.getUriForFile(this, "com.ubb.andrei.fileprovider", photoFile)
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, fileProvider)
            if (takePictureIntent.resolveActivity(this.packageManager) != null) {
                resultLauncherTakePhoto.launch(takePictureIntent)
                //startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
            } else {
                //startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
                resultLauncherTakePhoto.launch(takePictureIntent)
                Toast.makeText(this, "Unable to open camera", Toast.LENGTH_SHORT).show()
            }
        }

        btnOpenGallery?.setOnClickListener {
            val openGalleryIntent = Intent(Intent.ACTION_GET_CONTENT)
            openGalleryIntent.type = "image/*"

            resultLauncherOpenGallery.launch(openGalleryIntent)
            //startActivityForResult(openGalleryIntent, REQUEST_PICK_IMAGE)
        }
    }

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