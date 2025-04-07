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
    context = {"dialog": dialog_object}
    return render(request, "chat/dialog.html", context)

def analyze(request, pk):
    dialog_object = get_object_or_404(Dialog, pk=pk)
    label, explanation = get_analysis_from_model(dialog_object)
    dialog_object.label = label
    dialog_object.explanation = explanation
    dialog_object.save()

    context = {"dialog": dialog_object, "analyze": True}
    return render(request, "chat/dialog.html", context)

def get_analysis_from_model(dialog):
    sents = dialog.sentence_set.all()

    label = -1
    explanation = "Dummy Explanation for " + sents[0].content
    return (label, explanation)