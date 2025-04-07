from django.shortcuts import get_object_or_404, render
from django.views import generic

from .models import Role, Dialog, Sentence

class IndexView(generic.ListView):
    template_name = "chat/index.html"
    context_object_name = "dialog_list"

    def get_queryset(self):
        return Dialog.objects.all()

def dialog(request, pk):
    dialog_object = get_object_or_404(Dialog, pk=pk)
    return render(request, "chat/dialog.html", {"dialog": dialog_object})