from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login/',views.login,name='login'),
    path('fivebands/',views.fivebands,name='fivebands'),
    path('start/',views.start,name='start'),
    path('choose/',views.choose,name='choose'),
     path('navbar/',views.navbar,name='navbar'),


    path('alpha/',views.alpha,name='alpha'),
    path('theta/',views.theta,name='theta'),
    path('beta/',views.beta,name='beta'),
    path('delta/',views.delta,name='delta'),


     path('preprocess/',views.preprocess,name='preprocess'),
    path('features/',views.features,name='features'),
    path('classify/<str:jm>/',views.classify,name='classify'),
    path('result/',views.result,name='result'),

     path('delete/',views.delete,name='delete'),

   ] 
