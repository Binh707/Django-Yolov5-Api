import io
from PIL import Image as im
import torch

from django.shortcuts import render
from django.views.generic.edit import CreateView

from .models import ImageModel
from .forms import ImageUploadForm

print("un")
#path_hubconfig = "C:/Users/ASUS/Documents/GitHub/django_yolo_api/yolov5"
path_hubconfig = "yolov5"
#path_weightfile = "C:/Users/ASUS/Documents/GitHub/django_yolo_api/yolo5_weight/last.pt"  # or any custom trained model
path_weightfile = "yolo5_weight/yolov5s.pt"
model = torch.hub.load(path_hubconfig, 'custom',
                                   path=path_weightfile, source='local', force_reload=True)

class UploadImage(CreateView):
    model = ImageModel
    template_name = 'image/imagemodel_form.html'
    fields = ["image"]

    def post(self, request, *args, **kwargs):
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img = request.FILES.get('image')
            img_instance = ImageModel(
                image=img
            )
            img_instance.save()

            uploaded_img_qs = ImageModel.objects.filter().last()
            img_bytes = uploaded_img_qs.image.read()
            img = im.open(io.BytesIO(img_bytes))

            #path_hubconfig = "C:/Users/ASUS/Documents/GitHub/django_yolo_api/yolov5"
            #path_weightfile = "C:/Users/ASUS/Documents/GitHub/django_yolo_api/yolo5_weight/yolov5s.pt"  # or any custom trained model

            #model = torch.hub.load(path_hubconfig, 'custom',
                                   #path=path_weightfile, source='local', force_reload=True)

            results = model(img, size=640)
            results.render()
            for img in results.imgs:
                img_base64 = im.fromarray(img)
                img_base64.save("media/yolo_out/image0.jpg", format="JPEG")

            inference_img = "/media/yolo_out/image0.jpg"

            form = ImageUploadForm()
            context = {
                "form": form,
                "inference_img": inference_img
            }
            return render(request, 'image/imagemodel_form.html', context)

        else:
            form = ImageUploadForm()
        context = {
            "form": form
        }
        return render(request, 'image/imagemodel_form.html', context)
