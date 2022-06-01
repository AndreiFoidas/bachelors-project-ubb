package com.ubb.andrei

import android.graphics.Color
import android.graphics.Outline
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.AlarmClock.EXTRA_MESSAGE
import android.view.View
import android.view.ViewOutlineProvider
import android.view.Window
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.annotation.RequiresApi
import androidx.cardview.widget.CardView
import androidx.core.content.ContextCompat
import com.ubb.andrei.domain.plasticList

class InfoActivity : AppCompatActivity() {

    var recycleImageView: ImageView? = null
    var textName: TextView? = null
    var textDescription: TextView? = null
    var btnInfoBack: Button? = null
    var mainLayout: LinearLayout? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_info)
        val message = intent.getStringExtra(EXTRA_MESSAGE)

        recycleImageView = findViewById(R.id.recycleImageView)
        textName = findViewById(R.id.textName)
        textDescription = findViewById(R.id.textDescription)
        btnInfoBack  = findViewById(R.id.btnInfoBack)
        mainLayout = findViewById(R.id.mainLayout)


        val idFromMessage = message?.get(9)?.digitToInt()
        if (idFromMessage != null)
            if (idFromMessage in 1..8){
                val plastic = plasticList.data[idFromMessage - 1]

                textName?.text = plastic.name
                textDescription?.text = plastic.description
                recycleImageView?.setImageResource(plastic.photoPath)
                mainLayout?.setBackgroundColor(Color.parseColor(plastic.backgroundColor))
            }

        val card = findViewById<CardView>(R.id.eu)
        val curveRadius = 80F

        card.outlineProvider = object : ViewOutlineProvider() {

            @RequiresApi(Build.VERSION_CODES.LOLLIPOP)
            override fun getOutline(view: View?, outline: Outline?) {
                outline?.setRoundRect(0, 0, view!!.width, (view.height+curveRadius).toInt(), curveRadius)
            }
        }

        card.clipToOutline = true

        btnInfoBack?.setOnClickListener {
            onBackPressed()
        }

        val window: Window = this@InfoActivity.window
        // clear FLAG_TRANSLUCENT_STATUS flag:
        window.clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS)
        // add FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS flag to the window
        window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS)
        // finally change the color
        window.statusBarColor = ContextCompat.getColor(this@InfoActivity, R.color.black)
    }
}