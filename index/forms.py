from django import forms

class music_form(forms.Form):
    sound_file = forms.FileField(label='sound_file')
