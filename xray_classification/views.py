from xray_classification.scripts import *
from xray_classification.settings import MODEL_CL as model
from xray_classification.settings import img_size_cl as img_size
# from xray_classification.settings import streamer

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

def index():

    return('Success')

@csrf_exempt
def classify_api(request):
    data = {"success": False}

    if request.method == "POST":
        if request.FILES.get("image", None) is not None:
            image_request = request.FILES["image"]
            image_bytes = image_request.read()

        classify_result = get_prediction(model, image_bytes, img_size)

        if classify_result:
            data["success"] = True
            data["confidence"] = {}
            for idx, class_name in enumerate(classify_result["class_names"]):
                data["confidence"][class_name] = classify_result["scores"][idx]
            data["heatmap_path"] = classify_result["heatmap_url"]

    return JsonResponse(data)

# @csrf_exempt
# def batch_classify_api(request):
#     data = {"success": False}

#     if request.method == "POST":
#         if request.FILES.get("image", None) is not None:
#             image_request = request.FILES["image"]
#             image_bytes = image_request.read()

#         classify_result = streamer.predict(model, [image_bytes], img_size)[0]
#         # batch_prediction(model, image_bytes, img_size)

#         if classify_result:
#             data["success"] = True
#             data["confidence"] = {}
#             for idx, class_name in enumerate(classify_result["class_names"]):
#                 data["confidence"][class_name] = classify_result["scores"][idx]
#             data["heatmap_path"] = classify_result["heatmap_url"]

#     return JsonResponse(data)

