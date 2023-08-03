clc;
clear all;
close all;
%% Taking an Image
R = imread('100.jpg');
Re = imresize(R, [1304 3044]);
seg_img = rgb2gray(Re);


%% Feature Extraction
note=FeatureStatistical(seg_img);
c=input('Enter the Class');
try 
    load db
    note=[note c];
    db=[db; note];
    save db.mat db 
catch 
    db=[note c];
    save db.mat db
end



