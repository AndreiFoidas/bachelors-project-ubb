package com.ubb.andrei.domain

data class ServerResponse(
    val nr: Int,
    val name: String,
    val percentage: Double,
    val result: String,
    val filename: String
)
