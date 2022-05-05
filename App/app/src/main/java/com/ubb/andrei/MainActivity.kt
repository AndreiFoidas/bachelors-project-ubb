package com.ubb.andrei

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.ubb.andrei.domain.ServerResponse
import com.ubb.andrei.domain.plasticList
import com.ubb.andrei.utils.IObserver
import com.ubb.andrei.utils.URIPathHelper
import okhttp3.*
import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter


private const val FILE_NAME = "photo.jpg"
private lateinit var photoFile: File

class MainActivity : AppCompatActivity(), IObserver {

    private val REQUEST_PERMISSION = 10
    val pList = plasticList

    //Activity Main View
    var btnTakePicture: Button? = null
    var btnOpenGallery: Button? = null
    var btnUploadPicture: Button? = null
    var imageView: ImageView? = null

    var stringPhoto : String? = null
    var uriPhoto: Uri? = null
    var isTaken: Boolean? = null

     var photoGuess : ServerResponse? = null

    //Result Popup View
    var btnRight: Button? = null
    var btnWrong: Button? = null
    var btnBack: Button? = null
    var btnThis: Button? = null
    var icon: ImageView? = null
    var txtName: TextView? = null
    var txtRecyclable: TextView? = null
    var txtReusable: TextView? = null
    var plasticSpinner: Spinner? = null
    var spinnerAdapter: ArrayAdapter<String>? = null

    lateinit var dialogBuilder: AlertDialog.Builder
    lateinit var dialog: AlertDialog

    private fun checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.CAMERA),
                REQUEST_PERMISSION)
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
            //createNewResultPopup()
        }

        btnOpenGallery?.setOnClickListener {
            buttonOpenGalleryClicked()
        }

        btnUploadPicture?.setOnClickListener {

            buttonUploadPictureClicked()
        }

    }

    override fun onResume() {
        super.onResume()
        checkCameraPermission()
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

    private fun getPhotoFile(fileName: String): File {
        //Use 'getExternalFilesDir' on Context to access package-specific directories
        val storageDirectory = getExternalFilesDir(Environment.DIRECTORY_PICTURES)

        return File.createTempFile(fileName, ".jpg", storageDirectory)
    }

    fun buttonUploadPictureClicked(){
        val currentTime = LocalDateTime.now()
        val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss-SSS")
        val formatted = currentTime.format(formatter)
        val photoName = "$formatted.jpg"
        var result = ""

        if(isTaken == true) {
            UploadUtility(this@MainActivity, arrayListOf(this@MainActivity)).uploadFile(stringPhoto!!, photoName)
            //UploadUtility(this).uploadFile(stringPhoto!!)
        }
        if(isTaken == false){
            UploadUtility(this@MainActivity, arrayListOf(this@MainActivity)).uploadFile(uriPhoto!!, photoName)
            //UploadUtility(this).uploadFile(uriPhoto!!)
        }

        Log.d("C", photoGuess.toString())
        //createNewResultPopup()

    }

    fun createNewResultPopup(){
        dialogBuilder = AlertDialog.Builder(this)
        var resultPopupView = layoutInflater.inflate(R.layout.result_popup, null)
        var selectionSpinnerData = "SELECT PLASTIC"

        var spinnerData = arrayListOf<String>("SELECT PLASTIC", "1 PET", "2 HDPE", "3 PVC", "4 LDPE", "5 PP", "6 PS", "7 OTHER", "8 NOT PLASTIC")
        spinnerAdapter = ArrayAdapter(applicationContext, android.R.layout.simple_spinner_dropdown_item, spinnerData)
        spinnerAdapter?.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)

        icon = resultPopupView.findViewById<ImageView>(R.id.icon)
        txtName = resultPopupView.findViewById<TextView>(R.id.txtName)
        txtRecyclable = resultPopupView.findViewById<TextView>(R.id.txtRecyclable)
        txtReusable = resultPopupView.findViewById<TextView>(R.id.txtReusable)
        plasticSpinner = resultPopupView.findViewById<Spinner>(R.id.plasticSpinner)
        btnWrong = resultPopupView.findViewById<Button>(R.id.btnWrong)
        btnRight = resultPopupView.findViewById<Button>(R.id.btnRight)
        btnBack = resultPopupView.findViewById<Button>(R.id.btnBack)
        btnThis = resultPopupView.findViewById<Button>(R.id.btnThis)

        plasticSpinner?.adapter = spinnerAdapter
        plasticSpinner?.onItemSelectedListener = object:AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                //Toast.makeText(applicationContext, "You selected " + spinnerAdapter?.getItem(position), Toast.LENGTH_SHORT).show()
                selectionSpinnerData = spinnerAdapter?.getItem(position)!!
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                //TODO("Not yet implemented")
            }

        }

        btnWrong?.setOnClickListener {
            plasticSpinner?.visibility = View.VISIBLE
            btnWrong?.visibility = View.INVISIBLE
            btnRight?.visibility = View.INVISIBLE
            btnBack?.visibility = View.VISIBLE
            btnThis?.visibility = View.VISIBLE
        }

        btnRight?.setOnClickListener {
            if (photoGuess != null)
                UploadUtility(this@MainActivity, arrayListOf(this@MainActivity)).uploadRecyclingInfo(photoGuess?.name!!, photoGuess?.filename!!)
            // send to server
        }

        btnBack?.setOnClickListener {
            plasticSpinner?.visibility = View.INVISIBLE
            btnWrong?.visibility = View.VISIBLE
            btnRight?.visibility = View.VISIBLE
            btnBack?.visibility = View.INVISIBLE
            btnThis?.visibility = View.INVISIBLE
        }

        btnThis?.setOnClickListener {
            if (photoGuess != null && selectionSpinnerData != "SELECT PLASTIC" && selectionSpinnerData != "")
                UploadUtility(this@MainActivity, arrayListOf(this@MainActivity)).uploadRecyclingInfo(selectionSpinnerData, photoGuess?.filename!!)
            // send to server
        }

        plasticSpinner?.visibility = View.INVISIBLE

        Log.d("Z", photoGuess.toString())

        if (photoGuess != null){
            val id = photoGuess!!.nr

            if (id != -1) {
                var plastic = pList.data[id]

                Log.d("Z", plastic.toString())

                txtName?.text = "${plastic.number} ${plastic.abbreviation}"
                if (plastic.recyclable) {
                    txtRecyclable?.text = "Recyclable: YES"
                    txtRecyclable?.setTextColor(Color.GREEN)
                } else {
                    txtRecyclable?.text = "Recyclable: NO"
                    txtRecyclable?.setTextColor(Color.RED)
                }

                if (plastic.reusable) {
                    txtReusable?.text = "Reusable: YES"
                    txtReusable?.setTextColor(Color.GREEN)
                } else {
                    txtReusable?.text = "Reusable: NO"
                    txtReusable?.setTextColor(Color.RED)
                }
            }
        }

        dialogBuilder.setView(resultPopupView)
        dialog = dialogBuilder.create()
        dialog.show()
    }

    override fun update(guess: ServerResponse) {0
        photoGuess = guess
        Log.d("M", photoGuess.toString())
        this@MainActivity.runOnUiThread {
            createNewResultPopup()
        }
    }
}