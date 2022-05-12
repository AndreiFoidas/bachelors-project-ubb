package com.ubb.andrei.domain

import android.content.res.Resources
import com.ubb.andrei.R

data class Plastic (
    var number: Number,
    var abbreviation: String,
    var name: String,
    var description: String,
    var photoPath: Int,
    var reusable: Boolean,
    var recyclable: Boolean
)

object plasticList {
    val data: List<Plastic> = initPlastics()
}

fun initPlastics(): List<Plastic>{
    val plasticList: MutableList<Plastic> = emptyList<Plastic>().toMutableList()
    var plastic: Plastic
    //1 PET
    plastic = Plastic(1, "PET", "Polyethylene terephthalate",
    "PET is one of the most used types of plastic and is found in most water cans, but also in some packaging. It is intended for single use only, as repeated use increases " +
            "the risk of the release of harmful substances, sometimes even carcinogenic. Plastic of this type is difficult to disinfect, requiring dangerous chemicals.\n" +
            "PET is a type of recyclable plastic, in the process the plastic is crushed and then crushed into small flakes which are then reprocessed to create other products. \n\n" +
            "Category 1 (PET) plastic products should be recycled, but not reused.",
        R.drawable.p1, reusable = false, recyclable = true)
    plasticList.add(plastic)
    //2 HDPE
    plastic = Plastic(2, "HDPE", "High-density polyethylen",
            "HDPE is a rigid type of plastic from which detergent cans, toys and some bags are made, it is the most frequently recycled type of plastic and is considered one of the " +
            "least dangerous forms of plastic. This type of plastic is durable and not affected by direct sunlight, so it is used to make picnic tables, trash cans, benches and other " +
            "products that require durability. \n\nHDPE products are reusable and recyclable.",
        R.drawable.p2, reusable = true, recyclable = true)
    plasticList.add(plastic)
    //3 PVC
    plastic = Plastic(3, "PVC", "Polyvinyl chloride",
    "PVC is a type of soft and flexible plastic used for food packaging, for the production of toys for children and animals, computer cables and other products. This type of " +
            "plastic is also called \"poisonous plastic\" because it contains many toxins that it releases throughout its life cycle. \n\n" +
            "PVC products are not recyclable, and many of them are not reusable.",
        R.drawable.p3, reusable = false, recyclable = false)
    plasticList.add(plastic)
    //4 LDPE
    plastic = Plastic(4, "LDPE", "Low-density polyethylene",
    "LDPE is a type of plastic often found in heat-insulated packaging, in clothes bags from cleaners, in shopping bags, but also in some clothes and furniture. This type of " +
            "plastic is considered less toxic than others, but it is not normally recycled, as the products resulting from LDPE recycling are not very resistant. \n\n" +
            "LDPE products are reusable but not recyclable.",
        R.drawable.p4, reusable = true, recyclable = false)
    plasticList.add(plastic)
    //5 PP
    plastic = Plastic(5, "PP", "Polypropylene",
    "PP is a type of hard and light plastic that has excellent heat resistance qualities. It is used to pack food (to keep it fresh), to make lids for cans, margarine and " +
            "yogurt containers and straws. \n\n PP products are considered safe for reuse, but they are only recycled under certain conditions.",
        R.drawable.p5, reusable = true, recyclable = false)
    plasticList.add(plastic)
    //6 PS
    plastic = Plastic(6, "PS", "Polystyrene",
    "\n" +
            "PS is a cheap, lightweight plastic with a wide variety of uses: from disposable glasses and cutlery and egg boxes, to building materials. The products of this material " +
            "can release, especially as a result of their heating, carcinogenic substances. \n\n PS products should be avoided where possible, as recycling is quite difficult."
        , R.drawable.p6, reusable = false, recyclable = false)
    plasticList.add(plastic)
    //7 OTHER
    plastic = Plastic(7, "O", "Others",
    "\n" +
            "Category 7 plastic products are made from a combination of other types of plastic or less commonly used plastic formulas (such as BPA, polycarbonate and LEXAN) and are " +
            "difficult to recycle because they do not fall into a fixed category. This type of plastic is used to make products such as bottles and machine parts. \n\n" +
            "Products in this category are difficult to recycle and are not recommended for reuse.",
        R.drawable.p7, reusable = false, recyclable = false)
    plasticList.add(plastic)
    //8 NOT PLASTIC
    plastic = Plastic(8, "NOT PLASTIC", "Plastic combined with other materials",
        "", R.drawable.p8, reusable = false, recyclable = false)
    plasticList.add(plastic)

    return plasticList
}