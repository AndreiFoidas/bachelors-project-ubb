package com.ubb.andrei.utils

import com.ubb.andrei.domain.ServerResponse

interface IObserver {
    fun update(value: ServerResponse)
}

interface IObservable {
    val observers: ArrayList<IObserver>

    fun add(observer: IObserver) {
        observers.add(observer)
    }

    fun remove(observer: IObserver) {
        observers.remove(observer)
    }

    fun sendUpdateEvent(value: ServerResponse) {
        observers.forEach { it.update(value) }
    }
}