clc;
clear;
close all;

i=3;

if i==1
    R = imread('BASE_100.JPG');
[x,y,z] = size(R);
R_Gray = rgb2gray(R);
R_MARK1 = imcrop(R_Gray,[2130 0 600 250]);           
R_MARK2 = imcrop(R_Gray,[1940 300 280 230]);
R_MARK3 = imcrop(R_Gray,[2180 700 350 220]);  
R_MARK4 = imcrop(R_Gray,[2150 250 400 450]);
R_MARK5 = imcrop(R_Gray,[1400 1000 450 160]);
end

if i==2
    R = imread('BASE_500.JPG');
[x,y,z] = size(R);
R_Gray = rgb2gray(R);
R_MARK1 = imcrop(R_Gray,[2300 50 600 250]);           
R_MARK2 = imcrop(R_Gray,[2200 400 280 230]);
R_MARK3 = imcrop(R_Gray,[2430 800 350 220]);  
R_MARK4 = imcrop(R_Gray,[2400 350 400 450]);
R_MARK5 = imcrop(R_Gray,[1450 1100 450 160]);
end

if i==3
    R = imread('BASE_1000.JPG');
[x,y,z] = size(R);
R_Gray = rgb2gray(R);
R_MARK1 = imcrop(R_Gray,[2250 50 650 250]);           
R_MARK2 = imcrop(R_Gray,[2100 400 280 230]);
R_MARK3 = imcrop(R_Gray,[2300 830 400 220]);  
R_MARK4 = imcrop(R_Gray,[2300 350 400 500]);
R_MARK5 = imcrop(R_Gray,[1380 1050 540 160]);
end
        
N=imread('Test1000.jpg');
N = imresize(N, [x y]);
seg_img=rgb2gray(N);

a = min(R(:));
b = max(R(:));
R = (R-a).*(255/(b-a));
a = min(N(:));
b = max(N(:));
N = (N-a).*(255/(b-a));
 
%figure,subplot(211),imshow(R);
%title('Contrast stretched Real','fontsize',14);
%impixelinfo
%subplot(212),imshow(N);
%title('Contrast stretched Test','fontsize',14);
%impixelinfo

N_Gray = rgb2gray(N);
%figure,subplot(211),imshow(R_Gray);
%title('Grayscale image of the real note');
%subplot(212),imshow(N_Gray);
%title('Grayscale image of the test note');

if i==1
    N_MARK1 = imcrop(N_Gray,[2130 0 600 250]);           
    N_MARK2 = imcrop(N_Gray,[1940 300 280 230]);
    N_MARK3 = imcrop(N_Gray,[2180 700 350 220]);  
    N_MARK4 = imcrop(N_Gray,[2150 250 400 450]);
    N_MARK5 = imcrop(N_Gray,[1400 1000 450 160]);
end
if i==2
    N_MARK1 = imcrop(N_Gray,[2300 50 600 250]);           
    N_MARK2 = imcrop(N_Gray,[2200 400 280 230]);
    N_MARK3 = imcrop(N_Gray,[2430 800 350 220]);  
    N_MARK4 = imcrop(N_Gray,[2400 350 400 450]);
    N_MARK5 = imcrop(N_Gray,[1450 1100 450 160]);
end
if i==3
    N_MARK1 = imcrop(N_Gray,[2250 50 650 250]);           
    N_MARK2 = imcrop(N_Gray,[2100 400 280 230]);
    N_MARK3 = imcrop(N_Gray,[2300 830 400 220]);  
    N_MARK4 = imcrop(N_Gray,[2300 350 400 500]);
    N_MARK5 = imcrop(N_Gray,[1380 1050 540 160]);
end


figure,subplot(121),imshow(R_MARK1);
title('Real note');
subplot(122),imshow(N_MARK1);
title('Test note');

a = min(R_MARK2(:));
b = max(R_MARK2(:));
R_MARK2 = (R_MARK2-a).*(255/(b-a));
a = min(N_MARK2(:));
b = max(N_MARK2(:));
N_MARK2 = (N_MARK2-a).*(255/(b-a));

figure,subplot(121),imshow(R_MARK2);
title('Real note');
subplot(122),imshow(N_MARK2);
title('Test note');

figure,subplot(121),imshow(R_MARK3);
title('Real note');
subplot(122),imshow(N_MARK3);
title('Test note');

figure,subplot(121),imshow(R_MARK4);
title('Real note');
subplot(122),imshow(N_MARK4);
title('Test note');

a = min(N_MARK5(:));
b = max(N_MARK5(:));
N_MARK5 = (N_MARK5-a).*(255/(b-a));

figure,imshow(N_MARK5);
title('Contrast stretched Test Note');

sev = strel([0 1 2;-1 0 1;-2 -1 0]);
vertical_gradient_R = imdilate(R_MARK5,sev) - imerode(R_MARK5,sev);
vertical_gradient_N = imdilate(N_MARK5,sev) - imerode(N_MARK5,sev);
figure,subplot(211),imshow(vertical_gradient_R, []), title('Vertical gradient of Real image');
subplot(212),imshow(vertical_gradient_N, []), title('Vertical gradient of Test image');


stats = graycoprops(seg_img,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(seg_img);
Standard_Deviation = std2(seg_img);
Entropy = entropy(seg_img);
RMS = mean2(rms(seg_img));
Variance = mean2(var(double(seg_img)));
a = sum(double(seg_img(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(seg_img(:)));
Skewness = skewness(double(seg_img(:)));
m = size(seg_img,1);
n = size(seg_img,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = seg_img(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);

tnote = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];
%% Loading features 
Ftest = tnote;
load db.mat

[trainedClassifier, validationAccuracy] = trainClassifier(db)
v=predict(trainedClassifier.ClassificationSVM,tnote)

if v==1
    fprintf('Note 500 is real');
elseif v==2
    fprintf('Note 1000 is real');
elseif v==3
    fprintf('Note 500 is fake');
elseif v==4
    fprintf('Note 1000 is fake');
end

 
    
