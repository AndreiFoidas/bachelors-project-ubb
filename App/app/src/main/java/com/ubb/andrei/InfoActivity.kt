package com.ubb.andrei

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.AlarmClock.EXTRA_MESSAGE
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.ubb.andrei.domain.plasticList

class InfoActivity : AppCompatActivity() {

    var recycleImageView: ImageView? = null
    var textAbbreviation: TextView? = null
    var textName: TextView? = null
    var textDescription: TextView? = null
    var btnInfoBack: Button? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_info)
        val message = intent.getStringExtra(EXTRA_MESSAGE)

        recycleImageView = findViewById(R.id.recycleImageView)
        textAbbreviation = findViewById(R.id.textAbbreviation)
        textName = findViewById(R.id.textName)
        textDescription = findViewById(R.id.textDescription)
        btnInfoBack  = findViewById(R.id.btnInfoBack)

        var test = "Plastic: 0 TEST"
        val idFromMessage = message?.get(9)?.digitToInt()

        if (idFromMessage != null)
            if (idFromMessage in 1..8){
                val plastic = plasticList.data[idFromMessage - 1]

                textAbbreviation?.text = "${plastic.number} -  ${plastic.abbreviation}"
                textName?.text = plastic.name
                textDescription?.text = plastic.description
            }

        btnInfoBack?.setOnClickListener {
            onBackPressed()
        }
    }
}