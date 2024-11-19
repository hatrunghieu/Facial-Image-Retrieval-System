from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
from . import utils
from django.http import HttpResponse
from django.template import loader


def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            uploaded_file_url = fs.url(filename)

            # Process the uploaded image
            image_path = os.path.join(settings.MEDIA_ROOT, filename)
            processed_image = utils.process_image(image_path)

            # Extract embeddings
            embeddings = utils.extract_embeddings(processed_image)

            # Query the database
            results = utils.query_database(embeddings)

            return render(request, 'results.html', {'results': results, 'uploaded_file_url': uploaded_file_url})
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})


def testing(request):
    template = loader.get_template('template.html')
    context = {
        'uploaded_file_url': 'productionfiles/dataset/test/Abdullah_Gul/Abdullah_Gul_0011.jpg',
    }
    return HttpResponse(template.render(context, request))
