#!/usr/bin/env python
# coding=utf-8
'''
Description  : Implement singleton
Author       : unicornchan
Date         : 2024-06-11 17:08:36
Version      : 1.0.0
LastEditors  : chenxl 
LastEditTime : 2024-07-27 01:55:56
'''
import abc

class Singleton(abc.ABCMeta, type):
    """_summary_

    Args:
        abc.ABCMeta: Provide a mechanism for defining abstract methods and properties,
            enforcing subclasses to implement these methods and properties.
        type: Inherit from 'type' to make 'Singleton' a metaclass,
            enabling the implementation of the Singleton
    """
    _instances = {}

    def __call__(cls, *args, **kwds):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwds)
        return cls._instances[cls]

class AbstractSingleton(abc.ABC, metaclass=Singleton):
    """Provided an abstract Singleton base class, any class inheriting from
       this base class will automatically become a Singleton class.

    Args:
        abc.ABC: Abstract base class, it cannot be instantiated, only inherited. 
    """
