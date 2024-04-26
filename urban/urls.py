"""
URL configuration for urban project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from urban_app.views import *

urlpatterns = [
    path("admin/", admin.site.urls),

    path('',show_index),
    path('show_index/', show_index, name="show_index"), 
    path('check_login/', check_login, name="check_login"),
    path('logout/', logout, name="logout"),
    ##########################################
    path('show_home_admin/', show_home_admin, name="show_home_admin"), 
    path('show_emergency/', show_emergency, name="show_emergency"),
    path('show_meta_page/', show_meta_page, name="show_meta_page"),
    path('show_traffic_page/', show_traffic_page, name="show_traffic_page"),
    path('display_safety/', display_safety, name="display_safety"),
    path('display_test_page/', display_test_page, name="display_test_page"),
    path('upload_file/', upload_file, name="upload_file"),
    path('upload_metaimg/', upload_metaimg, name="upload_metaimg"),
    path('upload_traffic/', upload_traffic, name="upload_traffic"),

]
