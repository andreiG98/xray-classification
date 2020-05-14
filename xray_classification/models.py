from django.db import models

class User(models.Model):
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)

class Roles(models.Model):
    role_name = models.CharField(max_length=30)
    doctor = models.BooleanField

class Appointment(models.Model):
    creation_date = models.DateTimeField
    appointed_date = models.DateTimeField
