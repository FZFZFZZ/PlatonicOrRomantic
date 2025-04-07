from django.db import models

class Role(models.Model):
    name = models.CharField(default="UnknownPerson")
    number = models.IntegerField()

    def __str__(self):
        return self.name

class Dialog(models.Model):
    title = models.CharField()
    label = models.IntegerField()
    explanation = models.TextField()

    def __str__(self):
        return self.title

class Sentence(models.Model):
    role = models.ForeignKey(Role, on_delete=models.CASCADE)
    dialog = models.ForeignKey(Dialog, on_delete=models.CASCADE)
    content = models.TextField()

    def __str__(self):
        return self.content

