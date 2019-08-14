clc
clear 
close all

% import a image file and convert to a .wav for laser display
%image = imread('MH_test.jpg');
%image = imread('star.jpg');
%image = imread('star2.png');
image = imread('cloud2.png');
%image = imread('rec.png');

figure
subplot(3,2,1)
hold all
imshow(image)
title('Input Image')

% convert to grayscale 
I = rgb2gray(image);

subplot(3,2,2)
hold all
imshow(I)
title('Converted to grayscale')

% do a contor plot of the image
% use a single level to start with
%figure;
%C = imcontour(I,1);
C = contourc(double(I),1);

% split out to individual contours
index = 1;
i = 1;
while index < size(C,2)
    contour(i).level = C(1,index); %#ok<SAGROW>
    
    % read in the cords
    contour(i).x = C(1,index+(1:C(2,index))); %#ok<SAGROW>
    contour(i).y = C(2,index+(1:C(2,index)))*-1; %#ok<SAGROW>
           
    i = i + 1;
    index = index + C(2,index) + 1;
end
%contour(2:end) = contour(1);

max_size = max(size(image,1),size(image,2));
for n = 1:numel(contour)
    % move and scale between +- 1, maintain Aspect ratio
    % take size from original image
    contour(n).x = (contour(n).x / (0.5*max_size)) - 1;
    contour(n).y = (contour(n).y / (0.5*max_size)) + 1;

    % caculate the length, maybe also throw out by lenght, this is sort of
    % draw time i guess
    contour(n).length = sum(sqrt(diff(contour(n).x).^2 + diff(contour(n).y).^2));
    
    % start and end points of the contour
    contour(n).start = [contour(n).x(1),contour(n).y(1)];
    contour(n).end = [contour(n).x(2),contour(n).y(2)];
 end

% remove any that have zero lenght
contour([contour.length] == 0) = [];

% plot individual contors
subplot(3,2,[3,4,5,6])
title('Scalled and edges found')
hold all
xlim([-1,1])
ylim([-1,1])
axis equal
for n = 1:numel(contour)
    plot(contour(n).x,contour(n).y)
end


% conect the contors using greedy algotithum
% somthing more fancy here could do a better job
% also this also only considers joining up from the start and end
% could be some improment to be had by backtracing, or joining part way

% start at with the longest
[ ~, index] = max([contour.length]);
uesed = false(numel(contour),1);
sorted = zeros(numel(contour),1);
for n = 1:numel(contour)
    
    uesed(index) = true;
    sorted(n) = index;
    
    % find the diatance from this contors end to the start of all contours    
    starts = cell2mat({contour.start}');
        
    distance = inf(numel(contour),1);
    distance(~uesed) = sqrt(sum((starts(~uesed) - contour(index).end).^2,2));
    
    [ ~, index] = min(distance);
end

% make sure we have used all the contors, but only onece each
if ~all(uesed) || numel(unique(sorted)) ~= numel(sorted)
    error('sorting cock up')
end

% convert into single X Y data set
x_data = zeros(numel([contour.x]),1);
y_data = zeros(numel([contour.x]),1);
index = 1;
for n = 1:numel(contour)
    contour_size = numel([contour(sorted(n)).x]);
    
    x_data(index+(0:contour_size-1)) = contour(sorted(n)).x;
    y_data(index+(0:contour_size-1)) = contour(sorted(n)).y;
    
    index = index + contour_size;
end
% make joined up
x_data(end+1) = x_data(1);
y_data(end+1) = y_data(1);


% plot joined up contors
figure
subplot(6,4,[1,2])
hold all
plot(x_data)
ylabel('X data')
xlim([1,numel(x_data)])

subplot(6,4,[5,6])
hold all
plot(y_data)
ylabel('Y data')
xlim([1,numel(x_data)])

subplot(6,4,[3,4,7,8])
hold all
plot(x_data,y_data)
xlim([-1,1])
ylim([-1,1])
axis equal
title('Single line')


plot_time = 1/1;

%Sampling Frequency
Fs = 1/(plot_time/numel(x_data));


% do a low pass of 300hz
cutoff_freq = 300;

low_passed = zeros(length(x_data),2);
for i = 2:length(x_data)-1
    sample = [x_data(i),y_data(i)];
    
    rc = 1.0/(2*pi*cutoff_freq);
    
    alpha = (1/Fs)/((1/Fs)+rc);
    
    alpha = max(alpha,0);
    alpha = min(alpha,1);
    
    
    low_passed(i,:) = low_passed(i-1,:) + ((sample - low_passed(i-1,:)) * alpha);
end


subplot(6,4,[9,10])
hold all
plot(low_passed(:,1))
ylabel({'low passed';'X data'})
xlim([1,numel(low_passed(:,1))])

subplot(6,4,[13,14])
hold all
plot(low_passed(:,2))  
xlabel('samples')
ylabel({'low passed';'Y data'})
xlim([1,numel(low_passed(:,2))])

subplot(6,4,[11,12,15,16])
hold all
title(sprintf('%g Hz low pass filtered',cutoff_freq))
plot(low_passed(:,1),low_passed(:,2))
xlim([-1,1])
ylim([-1,1])
axis equal   

% change time step for maximum gradiat
% do a low pass of 300hz
max_change = mean(sqrt(diff(x_data(:,1)).^2  + diff(y_data(:,1)).^2));
time_strech = zeros(length(x_data),1);

for i = 2:length(x_data)
    change = sqrt((x_data(i)-x_data(i-1)).^2  + (y_data(i)-y_data(i-1)).^2);
    
    time_step = min((change / max_change),1);
    
    time_strech(i) =  time_strech(i-1) + time_step;
    

end

intep_time(:,1) = 1:0.1:time_strech(end);
x_data_resampled = interp1(time_strech,x_data,intep_time);
y_data_resampled = interp1(time_strech,y_data,intep_time);

out_hz = 100;

Fs_interp = out_hz*numel(intep_time);


subplot(6,4,[17,18])
hold all
plot(time_strech,x_data)
plot(x_data)
ylabel({'time strech';'X data'})
xlim(time_strech([1,end]))

subplot(6,4,[21,22])
hold all
plot(time_strech,y_data) 
plot(y_data)
xlabel('time')
ylabel({'time strech';'Y data'})
xlim(time_strech([1,end]))

subplot(6,4,[19,20,23,24])
hold all
title('time normalised')
plot(x_data,y_data)
xlim([-1,1])
ylim([-1,1])
axis equal   

num_repeat = 300;
file_name = sprintf('test_%i',plot_time);
audiowrite([file_name,'_orig.wav'],repmat([y_data*-1,x_data],num_repeat,1),round(Fs))
audiowrite([file_name,'.wav'],repmat([low_passed(:,2)*-1,low_passed(:,1)],num_repeat,1),round(Fs))
audiowrite([file_name,'_resampled.wav'],repmat([y_data_resampled*-1,x_data_resampled],num_repeat,1),round(Fs_interp))

    
