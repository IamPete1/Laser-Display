clc
clear 
close all

[input, sample_rate] = audioread('test_orig.wav');
%[input, sample_rate] = audioread('test.wav');
[input, sample_rate] = audioread('test_resampled2.wav');

%[input, sample_rate] = audioread('06 Planets.wav');

% only slows donw to min of 1000
% might need to resample
slow_down = 1;
sample_rate = round(sample_rate / slow_down);

% constrain to hard coded matlab limits
sample_rate = max(sample_rate,1000);
sample_rate = min(sample_rate,384000);

player = audioplayer(input,sample_rate);

plot_samples = 10000; % this is the sort of window of the track you see, also the sort of POV time for the laser

sample_time = (1-plot_samples:1:plot_samples) * sample_rate;
filler_data = zeros(plot_samples*2,2);

% Plot the channels serperatly
plot_xy = false;
if plot_xy
    figure
    subplot(2,1,1)
    hold all
    plot_1 = plot(sample_time,filler_data);
    plot([0,0],[1,-1],'--k')
    scatter_1 = scatter(0,0,'filled','k');
    xlim(sample_time([1,end]))
    ylim([-1,1])
    ylabel('CH 2, Right, Y')
    
    subplot(2,1,2)
    hold all
    plot_2 = plot(sample_time,filler_data);
    plot([0,0],[1,-1],'--k')
    scatter_2 = scatter(0,0,'filled','k');
    xlim(sample_time([1,end]))
    ylim([-1,1])
    ylabel('CH 1, Left, X')
    xlabel('Time (s), 0 is now')
else
    plot_1 = [];
    plot_2 = [];
    scatter_1 = [];
    scatter_2 = [];
end

% pretend to be a laser
% assume CH1 = left = X and CH2 = right = Y
figure 
hold all
title('Frickin  Laser')
col = linspace(0,1,10000)';
map = [ones(10000,1),col,col];
laser_plot = surface(zeros(plot_samples,2),zeros(plot_samples,2),zeros(plot_samples,2),zeros(plot_samples,2),...
            'facecol','no',...
            'edgecol','interp',...
            'linew',2);
colormap(map)
xlim([-1,1]) 
ylim([-1,1]) 
xlabel('CH 2, Right, X')
ylabel('CH 1, Left, Y')

start_time = rem(now,1);
player.TimerFcn = {@update_plots, plot_1, scatter_1, plot_2, scatter_2, input, sample_rate, plot_samples, start_time, laser_plot, plot_xy};
player.TimerPerio = 0.01; % this is the update rate of the plots in seconds, dont go too low or matlab will shit the bed, 0.05 = 20Hz
play(player);

function update_plots(player, ~, plot_1, scatter_1, plot_2, scatter_2, input, sample_rate, plot_samples, start_time, laser_plot, plot_xy)

    timeelapsed = (rem(now,1) - start_time)*24*3600; % time since start in seconds
    
    index = round(timeelapsed/(1/sample_rate)); % the index in the track of 'now'
    
    num_samples = size(input,1);
    
    % moving window
    index_start =  index - plot_samples;
    index_end =   index + plot_samples;
    
    plot_data = zeros(plot_samples*2,2);
    % attempt to be fancy at the start and end, completly unessisary
    if index_start < 1 && index_end > num_samples
        error('Track too short for sample rate')
    elseif index_start < 1
        % just starting out so 'scroll in' from the right
        index_start = 1;
        plot_data(plot_samples*2-(index_end-index_start):plot_samples*2,:) = input(index_start:index_end,:);
        
    elseif index_end > num_samples
        % finishing off so 'scroll out' to the left
        index_end = size(input,1);
        plot_data(1:(index_end-index_start),:) = input(index_start:index_end,:);
        
    else
        % normal
        plot_data = input(1+index_start:index_end,:);
    end
    
    laser_plot_data = nan(plot_samples,2);
    data = input(1+index_start:index,:);
    
    laser_plot_data(1:size(data,1),:) = data;
    
    col = linspace(1,0,plot_samples)';
    
    % update the plots
    try
        if plot_xy
            set(plot_1,'Ydata',plot_data(:,2))
            set(plot_2,'Ydata',plot_data(:,1))
            set(scatter_1,'Ydata',plot_data(plot_samples,2))
            set(scatter_2,'Ydata',plot_data(plot_samples,1))
        end
        
        % this is abit of odd way to do it as it also projects into the
        % future, but unless you care that it lines up with the audo it
        % makes no diffrence
        set(laser_plot,'Xdata',[laser_plot_data(:,1),laser_plot_data(:,1)])
        set(laser_plot,'Ydata',[laser_plot_data(:,2),laser_plot_data(:,2)])
        
        set(laser_plot,'CData',[col,col])

    catch
        % cant find plots, stop playing
        stop(player);
    end
    
    drawnow;
end