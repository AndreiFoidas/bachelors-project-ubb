package com.ubb.andrei

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.drawable.ColorDrawable
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.AlarmClock.EXTRA_MESSAGE
import android.provider.MediaStore
import android.util.Log
import android.view.*
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
    var btnMoreInfo: Button? = null
    var textPlasticMain: TextView? = null
    var resultLayout: LinearLayout? = null

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
    var linearLayoutButtons1: LinearLayout? = null
    var linearLayoutButtons2: LinearLayout? = null

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

        btnTakePicture = findViewById(R.id.btnTakePicture)
        btnOpenGallery = findViewById(R.id.btnOpenGallery)
        btnUploadPicture = findViewById(R.id.btnUploadPicture)
        imageView = findViewById(R.id.imageView)
        btnMoreInfo = findViewById(R.id.btnMoreInfo)
        textPlasticMain = findViewById(R.id.textPlasticMain)
        resultLayout = findViewById(R.id.resultLayout)

        btnTakePicture?.setOnClickListener {
            //buttonTakePictureClicked()
             //for testing
            photoGuess = ServerResponse(1, "Error", 0.0, "Fail", "")
            this@MainActivity.runOnUiThread {
                createNewResultPopup()
            }
        }

        btnOpenGallery?.setOnClickListener {
            buttonOpenGalleryClicked()
        }

        btnUploadPicture?.setOnClickListener {
            buttonUploadPictureClicked()
        }

        btnMoreInfo?.setOnClickListener {
            buttonMoreInfoClicked()
        }


        val window: Window = this@MainActivity.window
        // clear FLAG_TRANSLUCENT_STATUS flag:
        window.clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS)
        // add FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS flag to the window
        window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS)
        // finally change the color
        window.statusBarColor = ContextCompat.getColor(this@MainActivity, R.color.black)
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

        if(isTaken == true) {
            UploadUtility(this@MainActivity, arrayListOf(this@MainActivity)).uploadFile(stringPhoto!!, photoName)
        }
        if(isTaken == false){
            UploadUtility(this@MainActivity, arrayListOf(this@MainActivity)).uploadFile(uriPhoto!!, photoName)
        }
    }

    fun createNewResultPopup(){
        dialogBuilder = AlertDialog.Builder(this)
        var resultPopupView = layoutInflater.inflate(R.layout.result_popup, null)
        var selectionSpinnerData = "SELECT PLASTIC"

        var spinnerData = arrayListOf<String>("SELECT PLASTIC", "1 PET", "2 HDPE", "3 PVC", "4 LDPE", "5 PP", "6 PS", "7 OTHER", "8 NOT PLASTIC")
        //spinnerAdapter = ArrayAdapter(applicationContext, R.layout.spinner_item, spinnerData)
        val spinnerAdapter: ArrayAdapter<String?> = object :
            ArrayAdapter<String?>(applicationContext, R.layout.spinner_item,
                spinnerData as List<String?>
            ) {
            override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
                val v = super.getView(position, convertView, parent)
                (v as TextView)
                val nightModeFlags = context.resources.configuration.uiMode and
                        Configuration.UI_MODE_NIGHT_MASK
                when (nightModeFlags) {
                    Configuration.UI_MODE_NIGHT_YES -> {v.setTextColor(
                        resources.getColorStateList(R.color.alice_blue)
                    )}
                    Configuration.UI_MODE_NIGHT_NO -> {v.setTextColor(
                        resources.getColorStateList(R.color.smokey_black)
                    )}
                    Configuration.UI_MODE_NIGHT_UNDEFINED -> {}
                }

                return v
            }
        }
        spinnerAdapter.setDropDownViewResource(R.layout.spinner_dropdown_item)

        icon = resultPopupView.findViewById(R.id.icon)
        txtName = resultPopupView.findViewById(R.id.txtName)
        txtRecyclable = resultPopupView.findViewById(R.id.txtRecyclable)
        txtReusable = resultPopupView.findViewById(R.id.txtReusable)
        plasticSpinner = resultPopupView.findViewById(R.id.plasticSpinner)
        btnWrong = resultPopupView.findViewById(R.id.btnWrong)
        btnRight = resultPopupView.findViewById(R.id.btnRight)
        btnBack = resultPopupView.findViewById(R.id.btnBack)
        btnThis = resultPopupView.findViewById(R.id.btnThis)
        linearLayoutButtons1 = resultPopupView.findViewById(R.id.linearLayoutButtons1)
        linearLayoutButtons2 = resultPopupView.findViewById(R.id.linearLayoutButtons2)



        plasticSpinner?.adapter = spinnerAdapter
        plasticSpinner?.onItemSelectedListener = object:AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                selectionSpinnerData = spinnerAdapter.getItem(position)!!
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        btnWrong?.setOnClickListener {
            plasticSpinner?.visibility = View.VISIBLE
            linearLayoutButtons1?.visibility = View.GONE
            linearLayoutButtons2?.visibility = View.VISIBLE
        }

        btnRight?.setOnClickListener {
            if (photoGuess != null)
                UploadUtility(this@MainActivity, arrayListOf(this@MainActivity)).uploadRecyclingInfo(photoGuess?.name!!, photoGuess?.filename!!)
            resultLayout?.visibility = View.VISIBLE
            textPlasticMain?.text = "Plastic: ${photoGuess?.name!!}"
            dialog.dismiss()
        }

        btnBack?.setOnClickListener {
            plasticSpinner?.visibility = View.INVISIBLE
            linearLayoutButtons1?.visibility = View.VISIBLE
            linearLayoutButtons2?.visibility = View.GONE
        }

        btnThis?.setOnClickListener {
            if (photoGuess != null && selectionSpinnerData != "SELECT PLASTIC" && selectionSpinnerData != "")
                UploadUtility(this@MainActivity, arrayListOf(this@MainActivity)).uploadRecyclingInfo(selectionSpinnerData, photoGuess?.filename!!)
            resultLayout?.visibility = View.VISIBLE
            textPlasticMain?.text = "Plastic: $selectionSpinnerData"
            dialog.dismiss()
        }

        plasticSpinner?.visibility = View.INVISIBLE

        Log.d("Z", photoGuess.toString())

        if (photoGuess != null){
            val id = photoGuess!!.nr

            if (id != -1) {
                val plastic = pList.data[id]

                Log.d("Z", plastic.toString())

                txtName?.text = "${plastic.number} ${plastic.abbreviation}"
                icon?.setBackgroundResource(plastic.photoPath)
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
        dialog.window?.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
        dialog.show()
    }

    private fun buttonMoreInfoClicked() {
        val message = textPlasticMain?.text
        val intent = Intent(this, InfoActivity::class.java).apply {
            putExtra(EXTRA_MESSAGE, message)
        }
        startActivity(intent)
    }

    override fun update(guess: ServerResponse) {
        photoGuess = guess
        Log.d("M", photoGuess.toString())
        this@MainActivity.runOnUiThread {
            createNewResultPopup()
        }
    }
}