from django.db import models

# Create your models here.
class video_player(models.Model):
    # name = models.TextField()
    # file = models.FileField(upload_to='videos/')
    title = models.CharField(max_length=100)
    description = models.TextField()
    video = models.FileField(upload_to='videos/')
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title