clc
clear 
close all

% import a image file and convert to a .wav for laser display
image = imread('MH_test.jpg');
%image = imread('star.jpg');
%image = imread('star2.png');
%image = imread('cloud2.png');
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

    contour(n) = simplify_contour(contour(n));

    % caculate the length, maybe also throw out by lenght, this is sort of
    % draw time i guess
    contour(n).length = sum(sqrt(diff(contour(n).x).^2 + diff(contour(n).y).^2));
    
    % start and end points of the contour
    contour(n).start = [contour(n).x(1),contour(n).y(1)];
    contour(n).end = [contour(n).x(end),contour(n).y(end)];
end

% remove any that have short lenght
contour([contour.length] < 10^-2) = [];

% plot individual contors
subplot(3,2,[3,4,5,6])
title('Scalled and edges found')
hold all
xlim([-1,1])
ylim([-1,1])
axis equal
for n = 1:numel(contour)
    plot(contour(n).x,contour(n).y,'*-')
end


% conect the contors using greedy algotithum
% somthing more fancy here could do a better job
% also this also only considers joining up from the start and end
% could be some improment to be had by backtracing, or joining part way??

% start at with the longest
[ ~, index] = max([contour.length]);
uesed = false(numel(contour),1);
sorted = zeros(numel(contour),1);
% find the diatance from this contors end to the start of all contours    
starts = cell2mat({contour.start}');
for n = 1:numel(contour)
    
    uesed(index) = true;
    sorted(n) = index;
            
    distance = inf(numel(contour),1);
    distance(~uesed) = sqrt(sum((starts(~uesed) - contour(index).end).^2,2));
    
    [ ~, index] = min(distance);
end

% two-opt tsp
[sorted, reversed] = two_opt(starts,cell2mat({contour.end}'),cell2mat({contour.length}'));


% make sure we have used all the contors, but only onece each
if ~all(uesed) || numel(unique(sorted)) ~= numel(sorted)
    error('sorting cock up')
end

% see if we can move the start or end point round
% closed = true(numel(contour),1);
% for i = 2:numel(contour)-1
%     if all(contour(i).start == contour(i).end) 
%         % this is a closed contor
%         if ~reversed(i-1)
%            prev_point =  contour(i-1).end;
%         else
%            prev_point =  contour(i-1).start;
%         end
%         
%         if ~reversed(i+1)
%            next_point =  contour(i+1).start;
%         else
%            next_point =  contour(i+1).end;
%         end
%         
%         % find the min distange
%         length_contour = numel(contour(i).x);
%         dist = zeros(numel(contour(i).x),1);
%         for j = 1:length_contour
%             dist(j) = sqrt((contour(i).x(j) - prev_point(1))^2 + (contour(i).y(j) - prev_point(2))^2) + ...
%                       sqrt((contour(i).x(j) - next_point(1))^2 + (contour(i).y(j) - next_point(2))^2);
%         end
%         
%         [~,index] = min(dist);        
%         contour(i).x = contour(i).x([index:length_contour-1,1:index]);    
%         contour(i).y = contour(i).y([index:length_contour-1,1:index]);
%     end
% end

% convert into single X Y data set
x_data = zeros(numel([contour.x]),1);
y_data = zeros(numel([contour.x]),1);
index = 1;
for n = 1:numel(contour)
    contour_size = numel([contour(sorted(n)).x]);
    
    if ~reversed(n)
        x_data(index+(0:contour_size-1)) = contour(sorted(n)).x;
        y_data(index+(0:contour_size-1)) = contour(sorted(n)).y;
    else
        x_data(index+(0:contour_size-1)) = fliplr(contour(sorted(n)).x);
        y_data(index+(0:contour_size-1)) = fliplr(contour(sorted(n)).y);
    end
    
    index = index + contour_size;
end
% make joined up
x_data(end+1) = x_data(1);
y_data(end+1) = y_data(1);

% work out the total cost
cost = sum(sqrt(diff(x_data).^2 + diff(y_data).^2))

% plot joined up contors
figure
%subplot(6,4,[1,2])
subplot(2,4,[1,2])

hold all
plot(x_data)
ylabel('X data')
xlim([1,numel(x_data)])

%subplot(6,4,[5,6])
subplot(2,4,[5,6])

hold all
plot(y_data)
ylabel('Y data')
xlim([1,numel(x_data)])

%subplot(6,4,[3,4,7,8])
subplot(2,4,[3,4,7,8])
hold all
plot(x_data,y_data)
xlim([-1,1])
ylim([-1,1])
axis equal
title('Single line')


plot_time = 1/2;

%Sampling Frequency
Fs = 192000; %1/(plot_time/numel(x_data));

%{
% % do a low pass of 300hz
% cutoff_freq = 300;
% 
% low_passed = zeros(length(x_data),2);
% for i = 2:length(x_data)-1
%     sample = [x_data(i),y_data(i)];
%     
%     rc = 1.0/(2*pi*cutoff_freq);
%     
%     alpha = (1/Fs)/((1/Fs)+rc);
%     
%     alpha = max(alpha,0);
%     alpha = min(alpha,1);
%     
%     
%     low_passed(i,:) = low_passed(i-1,:) + ((sample - low_passed(i-1,:)) * alpha);
% end
% 
% 
% subplot(6,4,[9,10])
% hold all
% plot(low_passed(:,1))
% ylabel({'low passed';'X data'})
% xlim([1,numel(low_passed(:,1))])
% 
% subplot(6,4,[13,14])
% hold all
% plot(low_passed(:,2))  
% xlabel('samples')
% ylabel({'low passed';'Y data'})
% xlim([1,numel(low_passed(:,2))])
% 
% subplot(6,4,[11,12,15,16])
% hold all
% title(sprintf('%g Hz low pass filtered',cutoff_freq))
% plot(low_passed(:,1),low_passed(:,2))
% xlim([-1,1])
% ylim([-1,1])
% axis equal   
% 
% % change time step for maximum gradiat
% % do a low pass of 300hz
% max_change = mean(sqrt(diff(x_data(:,1)).^2  + diff(y_data(:,1)).^2));
% time_strech = zeros(length(x_data),1);
% 
% for i = 2:length(x_data)
%     change = sqrt((x_data(i)-x_data(i-1)).^2  + (y_data(i)-y_data(i-1)).^2);
%     
%     time_step = min((change / max_change),1);
%     
%     time_strech(i) =  time_strech(i-1) + time_step;
%     
% 
% end
% 
% intep_time(:,1) = 1:0.1:time_strech(end);
% x_data_resampled = interp1(time_strech,x_data,intep_time);
% y_data_resampled = interp1(time_strech,y_data,intep_time);
% 
% out_hz = 25;
% 
% Fs_interp = Fs;%out_hz*numel(intep_time);
% 
% 
% subplot(6,4,[17,18])
% hold all
% plot(time_strech,x_data)
% plot(x_data)
% ylabel({'time strech';'X data'})
% xlim(time_strech([1,end]))
% 
% subplot(6,4,[21,22])
% hold all
% plot(time_strech,y_data) 
% plot(y_data)
% xlabel('time')
% ylabel({'time strech';'Y data'})
% xlim(time_strech([1,end]))
% 
% subplot(6,4,[19,20,23,24])
% hold all
% title('time normalised')
% plot(x_data,y_data)
% xlim([-1,1])
% ylim([-1,1])
% axis equal   
% 
% num_repeat = 500;
% %file_name = sprintf('test_%i',plot_time);
% file_name = 'test';
% %audiowrite([file_name,'_orig.wav'],repmat([x_data,y_data,],num_repeat,1),round(Fs))
% %audiowrite([file_name,'.wav'],repmat([low_passed(:,1),low_passed(:,2)],num_repeat,1),round(Fs))
% %audiowrite([file_name,'_resampled.wav'],repmat([x_data_resampled,y_data_resampled],num_repeat,1),round(Fs_interp))
%}

num_repeat = 500;
file_name = 'test';

% basic resample to get correct number of display hz
disp_hz = 10;

% how many samples do we get to draw the whole thing in?
total_samples = Fs * (1/disp_hz);

if numel(x_data) > total_samples
    orig_time = 1:numel(x_data);
    
    new_time = linspace(1,numel(x_data),total_samples);
    
    x_resample2 = interp1(orig_time,x_data',new_time)';
    y_resample2 = interp1(orig_time,y_data',new_time)';
else
    x_resample2 = x_data;
    y_resample2 = y_data;
end

audiowrite([file_name,'_resampled2.wav'],repmat([x_resample2,y_resample2],num_repeat,1),round(Fs))


function [Contour_tour, reversed] = two_opt(starts, ends, lengths)

points = [starts; ends;];
points_contor = [(1:size(starts,1))';(1:size(ends,1))'];

tour_length = size(points,1);

% build a cost matrix
cost_mat = inf(tour_length);
for i = 1:tour_length
    for j = 1:tour_length
        if i == j
            continue
        end        
        if points_contor(i) == points_contor(j)
             cost_mat(i,j) = lengths(points_contor(i));
             continue
        end
        cost_mat(i,j) = sqrt( (points(i,1) - points(j,1))^2 + (points(i,2) - points(j,2))^2 );
    end
end

% work out the cost
cost = 0;
for i = 1:tour_length -1 
    cost = cost + cost_mat(i,i+1);
end
cost = cost + cost_mat(end,1);

fprintf('Start Cost %g\n',cost)

% Greedy Start
tour = ones(tour_length,1);
i = 1;
while i < tour_length   
    
    index = find(points_contor == points_contor(tour(i)));
    index(index == tour(i)) = [];
    
    i = i + 1;
    tour(i) = index;
    
    if i < tour_length
        costs = cost_mat( tour(i),:);
        costs(unique(tour)) = inf;
        
        [~,index] = min(costs);
        i = i + 1;
        tour(i) = index;
    end
end


% work out the cost
cost = 0;
for i = 1:tour_length -1 
    cost = cost + cost_mat(tour(i),tour(i+1));
end
cost = cost + cost_mat(tour(end),tour(1));

fprintf('Greedy Cost %g\n',cost)

max_iter = 1500;
for iter = 1:max_iter
    % try all swaps
    improved = false;
    for i = 1:2:tour_length
        for j = i+1:2:tour_length
           
            start_bit = tour(1:i-1);
            middle_bit = tour(i:j);
            end_bit = tour(j+1:end);
            
            test_route = [start_bit; flipud(middle_bit); end_bit;];
            
            test_cost = 0;
            for k = 1:tour_length -1
                test_cost = test_cost + cost_mat(test_route(k),test_route(k+1));
            end
            test_cost = test_cost + cost_mat(test_route(end),test_route(1));
            
            
            if test_cost < cost
                tour = test_route;
                cost = test_cost;
                improved = true;
                break;
            end
        end
        if improved
            break
        end
    end
    if ~improved
        break;
    end
end

if iter == max_iter
    warning('Two-Opt Max itter')
end

fprintf('Tow-opt Cost %g\n',cost)

% sort back into contor order
tour = reshape(tour,[2,numel(tour)/2])';
Contour_tour = points_contor(tour);

if any(Contour_tour(:,1) ~= Contour_tour(:,2))
    error('cock up')
end
    

Contour_tour = Contour_tour(:,1);
reversed = tour(:,1) > tour(:,2);

end

function contour = simplify_contour(contour)

% figure
% hold all
% title(sprintf('before %i points',numel(contour.x)))
% plot(contour.x,contour.y,'*-')

i = 2;
while i < numel(contour.y) - 1
    dy = abs(diff(contour.y([i-1,i,i+1])));
    dx = abs(diff(contour.x([i-1,i,i+1])));
    
    ang = atan2d(dy,dx);
    line_length = sqrt(dy.^2 + dx.^2);
    change = diff(ang);
    
    if  change < -180
        change = change + 360;
    elseif change > 180
        change = change - 360;
    end
    
    
    
    if abs(change) < 5 || any(line_length < 0.001)
        contour.x(i) = [];
        contour.y(i) = [];
    else
        i = i + 1;
    end
end

% figure
% hold all
% title(sprintf('after %i points',numel(contour.x)))
% plot(contour.x,contour.y,'*-')


end





    
