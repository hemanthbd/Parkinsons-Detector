%% Extracting Features for Healthy Data

clc;
clear all;
close all;

Audio_files_1 = dir('C:/Users/User1/Documents/MATLAB/Project_Phonations/Our_Dataset/Healthy_files/*.wav');
filename2 = 'final_jit_shim_values_healthy.csv';
Jitt_Shimm_healthy = csvread(filename2,2,2);
%% For 2000 frames each 

for i=1:50
    patient_no = i;
    X = sprintf('Calculating features for audio sample of Healthy patient %d',patient_no);
    disp(X)
    [y{i}, Fs{i}] = audioread(Audio_files_1(i).name); % Reading the Audio1 files
    %disp(Fs{i})
    %[ym{i}, Fsm{i}] = audioread(Audio_files_2{i}); % Reading the  Audio2 files
     [x_overlap, fs] = overlap_frame(y{i},Fs{i}); 
     [~,col]= size(x_overlap);
     %disp(col)
     
%%
     spl = splnorm(x_overlap,fs); 
     %% Bark-Band Features-> 145 Features
     energy_barkband = barkband(spl);
     [kurtosis_barkband, skew_barkband] = kurskewbark(spl);
     [iqr_barkband, mad_barkband, std_barkband] = spreadbark(spl);
     % Combining
     f1 = reshape(energy_barkband,[1,25*col]);
     f2 = reshape(kurtosis_barkband,[1,24*col]);
     f3 = reshape(skew_barkband,[1,24*col]);
     f4 = reshape(iqr_barkband,[1,24*col]);
     f5 = reshape(mad_barkband,[1,24*col]);
     f6 = reshape(std_barkband,[1,24*col]);
     combo_bark = [f1 f2 f3 f4 f5 f6];
     %% Statistics-> 347*6 Features
     crest_db = crest(x_overlap);
     [kurtosis_sig, skew_sig] = kurskew_sig(x_overlap);
     [iqr_sig, mad_sig, std_sig] = spread_sig(x_overlap);
     combo_stat = [crest_db kurtosis_sig skew_sig iqr_sig mad_sig std_sig];
     
    
     %%  GFCC Features -> 76 Features
     % GFCC-> Double-Delta GFCC-> 26 + 26 + 26 Features
     gfcc_sig = gfcc(x_overlap); 
    % GFCC Delta
     gfcc12_delta=zeros(26,col);
    for k=2:26
        gfcc12_delta(k,:)=gfcc_sig(k,:)-gfcc_sig(k-1,:);
    end
    % GFCC Double-Delta
    gfcc12_doubledelta = zeros(26,col);
    for k=2:26
        gfcc12_doubledelta(k,:)=gfcc12_delta(k,:)-gfcc12_delta(k-1,:);
    end
   g1 = reshape(gfcc_sig,[1,26*col]);
   g2 = reshape(gfcc12_delta,[1,26*col]);
   g3 = reshape(gfcc12_doubledelta,[1,26*col]);
   combo_gfcc = [g1 g2 g3];
    %% MFCC Features 78 Features
%      % MFCC,Delta, Double-Delta MFCC-> 26 + 26 + 26 Features
      [mfcc26 ,mfcc26_delta, mfcc26_doubledelta] = mfcc(x_overlap);
      mfcc1 = reshape(mfcc26, [1, 26*col]);
      mfcc2 = reshape(mfcc26_delta, [1, 26*col]);
      mfcc3 = reshape(mfcc26_doubledelta, [1, 26*col]);
      combo_mfcc = [mfcc1 mfcc2 mfcc3];

     %%
     % Pitch
    pitch_sig =pitch(x_overlap);
    pitch_sig(~isfinite(pitch_sig))=0;
     % Pitch Salience
     % Pitch Instant. Confidence
    combo_pitch = [pitch_sig];
     %% Spectral
%      % Spectral Flatness
      spec_flat= spflat(x_overlap);
      spec_energy = senergy(x_overlap);
      spec_rolloff = srolloff(x_overlap);
      spec_centroid = scentroid(x_overlap);
      spec_zr = zerocross(x_overlap);
      spec_flux = sflux(x_overlap);
%      spec_lef = slowenergy(x_overlap);
       combo_spec = [spec_flat spec_energy spec_rolloff spec_centroid spec_zr spec_flux];
     %% Loudness
     loud_x = loud(y{i});
     combo_loud = [loud_x];
     %% LPC Coefficients
      lpcoeff = lpcf(x_overlap);
      combo_lpc = reshape(lpcoeff, [1, 26*col]);
     %% Concatenating all Features
     combo_all{i} = [combo_bark combo_gfcc combo_mfcc combo_stat combo_pitch combo_spec combo_lpc combo_loud Jitt_Shimm_healthy(i,:)];
     %disp(length(combo_all{i}))
     train_input_H(i,:) = combo_all{i};
end
save('train_input_H.mat','train_input_H')

    f=1:256;
    f_bark = freq2bark(f,0);
    %Tq2 = real(10*log10(tq(f)));
%     figure;
%     plot(f,r1(1:257,60));
%     xlabel('Frequency (Hz)')
%     ylabel('SPL (dB)');
%     title('PSD- SPL Normalized')
%     
%     figure;
%     plot(f_bark,r1(1:256,frame_num));
%     xlabel('Bark Frequency (z)')
%     ylabel('SPL (dB)');
%     title('Step1: PSD- SPL Normalized')
    

%%
function [overf, F_s] = overlap_frame(X,fs)
%disp(fs);
n=512; %how many samples will each frame contain
Fs = 44100;
%disp(fs)
if fs~=44100
    r = resample(X(:,1),160,441);
else
    r = X(:,1);
end
preemph = [1 -0.97];
r = filter(1,preemph,r);
r=r-mean(r); % remove DC component
s=r/max(abs(r));  %normalization
x = s;
n_overlap_frames = floor((length(x)-n)/(n/2));
x_overlap = zeros(512,n_overlap_frames);
x_hamm=zeros(512,n_overlap_frames);

for k=0:n_overlap_frames
    x_overlap(:,k+1)=x(1+(n*k/2):n*(k+1)-((k*n)/2));
    x_hamm(:,k+1)= hamming(length(x_overlap(:,k+1))).*x_overlap(:,k+1);
end
%disp(n_overlap_frames)
overf =  x_hamm(:,100:2100);
F_s = fs;
end

%%
function snorm = splnorm(X,fs)
[~, cols] = size(X);
n_overlap_frames =cols;
if fs~=44100
    X =resample(X,160,441);
end
N=512;
b = 16;
%X = (X - mean(X));
%X = X./max(abs(X));
%X = X/(N*2^(b-1));
x_fft = zeros(512,n_overlap_frames);
psd = zeros(512,n_overlap_frames);
P = zeros(256,n_overlap_frames);
PN=90.302;
%disp(n_overlap_frames-1);
for k=0:n_overlap_frames-1
    %disp(k)
    x_fft(:,k+1)= fft(X(:,k+1),512);
    psd(:,k+1)= (abs(x_fft(:,k+1)).^2);
    %psd(2:end-1,k+1)= 2*psd(2:end-1,k+1);
    P(:,k+1)= PN + 10*log10(psd(1:256,k+1));
end
snorm =  P;

end

%% 
function bb= barkband(X)

[~, cols] = size(X);
n_frames = cols;
bw = [0 100 200 300 400 510 630 770 920 1080 1270 1480 1720 2000 2320 2700 3150 3700 4400 5300 6400 7700 9500 12000 15500 22050];
f=1:256;
bark = freq2bark(f,0);
energy_bb = zeros(length(bw)-1,n_frames);
%disp(length(bw))
for k=1:n_frames
    %disp(k);
    for j=1:length(bw)-1
        if j==1
            l=1;
        else
            l = round(bw(j)/44100*512);
        end
        %disp(j);
        %disp(l);        
        u = round(bw(j+1)/44100*512); 
        %disp(u);
        energy_bb(j,k)= rms(X(l:u,k)).^2; 
    end  
end
bb = energy_bb;
end

%%
function [kurbb, skeww] = kurskewbark(X)
[~, cols] = size(X);
n_frames = cols;
bw = [0 100 200 300 400 510 630 770 920 1080 1270 1480 1720 2000 2320 2700 3150 3700 4400 5300 6400 7700 9500 12000 15500 22050];
f=1:256;
%bark = freq2bark(f,0);
kurt_bb = zeros(length(bw)-2,n_frames);
skew_bb = zeros(length(bw)-2,n_frames);

%disp(length(bw))
for k=1:n_frames
    %disp(k);
    for j=1:length(bw)-1
        if j==1
            l=1;
        else
            l = round(bw(j)/44100*512);
        end
        u = round(bw(j+1)/44100*512); 
        %disp(u);
        kurt_bb(j,k)= kurtosis(X(l:u,k));
        skew_bb(j,k)= skewness(X(l:u,k));
    end  
end
kurbb = kurt_bb(2:end,:);
skeww = skew_bb(2:end,:);

end

%%
function crestdb = crest(X)
[~, cols] = size(X);
n_frames = cols;

crest_db = zeros(1,n_frames);

%disp(length(bw))
for k=1:n_frames
    %disp(k);
   crest_db(1,k)= 20*log10(max(findpeaks(xcorr(X(:,k))))/rms(X(:,k)));
      
end

crestdb = crest_db;

end

%%
function [kurtosis_sig, skew_sig] = kurskew_sig(X)
[~, cols] = size(X);
n_frames = cols;

kurtosis1 = zeros(1,n_frames);
skew1 = zeros(1,n_frames);

%disp(length(bw))
for k=1:n_frames
    %disp(k);
   kurtosis1(1,k)= kurtosis(X(:,k));
   skew1(1,k) = skewness(X(:,k));  
end

kurtosis_sig = kurtosis1;
skew_sig = skew1;
end
%%
function [iqr1,mad1,std1]= spreadbark(X)
[~, cols] = size(X);
n_frames = cols;
bw = [0 100 200 300 400 510 630 770 920 1080 1270 1480 1720 2000 2320 2700 3150 3700 4400 5300 6400 7700 9500 12000 15500 22050];
f=1:256;
bark = freq2bark(f,0);
iqr_bb = zeros(length(bw)-1,n_frames);
mad_bb = zeros(length(bw)-1,n_frames);
std_bb = zeros(length(bw)-1,n_frames);

%disp(length(bw))
for k=1:n_frames
    %disp(k);
    for j=1:length(bw)-1
        if j==1
            l=1;
        else
            l = round(bw(j)/44100*512);
        end
        %disp(j);
        %disp(l);        
        u = round(bw(j+1)/44100*512); 
        %disp(u);
        iqr_bb(j,k)= iqr(X(l:u,k));
        mad_bb(j,k)= mad(X(l:u,k));
        std_bb(j,k)= std(X(l:u,k));
    end  
end
iqr1 = iqr_bb(2:end,:);
mad1 = mad_bb(2:end,:);
std1 = std_bb(2:end,:);
end

%%
function [iqr_sig, mad_sig, std_sig] = spread_sig(X)
[~, cols] = size(X);
n_frames = cols;

iqr_bb = zeros(1,n_frames);
mad_bb = zeros(1,n_frames);
std_bb = zeros(1,n_frames);

%disp(length(bw))
for k=1:n_frames
        iqr_bb(1,k)= iqr(X(:,k));
        mad_bb(1,k)= mad(X(:,k));
        std_bb(1,k)= std(X(:,k));
end
iqr_sig = iqr_bb;
mad_sig = mad_bb;
std_sig = std_bb;
end

%% 
function b = freq2bark(freq_bins,flag)
fs = 44100;
freq_arr = fs*freq_bins/512;
bark = zeros(1,length(freq_arr));
for i=1:length(bark)
    if freq_arr(i)<=1500
        bark(i)=13*atan(0.76*freq_arr(i)/1000) + 3.5*atan((freq_arr(i)/7500).^2);
    else
        bark(i)=8.7 + 14.2*log10(freq_arr(i)/1000);
        
    end
end
if flag==1
    b = round(bark);
else
    b= (bark);
end
end
%%
function lpcoeff = lpcf(X)
[~, cols] = size(X); % Overlapped + Hamming Windowed Frames
n_overlap_frames =cols;
lpc26 = zeros(26,n_overlap_frames);
for k=1:n_overlap_frames
    lpc26(:,k) = lpc(X(:,k),25);
end
lpcoeff = lpc26;

end
%%
function gfcc_s = gfcc(X)
[~, cols] = size(X); % Overlapped + Hamming Windowed Frames
n_overlap_frames =cols;
x_fft = zeros(512,n_overlap_frames);
gfcc13 = zeros(26,n_overlap_frames);
%disp(n_overlap_frames-1);
for k=0:n_overlap_frames-1
    %disp(k)
    % Windowing
    % FFT
    x_fft(:,k+1)= fft(X(:,k+1),512);
    % Make GT Filters
end
 
 gm2 = zeros(64,327);
 
 for e=1:cols
     x3=x_fft(:,e);
     x_h = X(:,e);
     fcfs = MakeERBFilters(44100,64,100);
      fb = ERBFilterBank(x_h', fcfs);
%       disp(size(gm))
%       disp(size(x3))
      gm2(:,e)= fb*x3;
 end
    % Find the Logarithm 
    lg = log10(gm2);
    % GFCC 13 Coefficients from DCT (2-14)
    gfcc64 = abs(dct(lg));
    gfcc13(:,:)= gfcc64(2:27,:); 
    gfcc_s = gfcc13;
end

function [mfcc_sig,mfcc_delta, mfcc_dd] = mfcc(X)
[~, cols] = size(X); % Overlapped Frames
n_overlap_frames =cols;
dft = zeros(257,n_overlap_frames);
for k=1:cols
    p = periodogram(X(:,k),[],512);
    dft(:,k) = p;    
end
Fs = 44100;
% Mel-Filter Bank %

n_filt=40;
fl=0;
fH=Fs/2;
fl_mel=0;
fH_mel = 2595*log10(1+(fH/700));
mel_points = zeros(42,1);
inc = (fH_mel-fl)/(n_filt+1);
init=0;
mel_points(1) = fl_mel;
for i=1:42
    mel_points(i)=init;
    init=init+inc;
end

hz_points=zeros(42,1);
for i=1:42
    hz_points(i)=700*(10^(mel_points(i)/2595)-1);
end

bin_hz = zeros(42,1);
for i=1:42
    bin_hz(i)= floor((512 + 1) * hz_points(i)/Fs)+1;
end

fbanks = zeros(n_filt,(floor(256 / 2 + 1)));

for i=2:41
    f_left = bin_hz(i-1);
    f_center = bin_hz(i);
    f_right = bin_hz(i+1);
    
    for j=f_left:f_center
       fbanks(i-1,j)= (j- bin_hz(i-1))/(bin_hz(i)-bin_hz(i-1));  
    end
    
    for j=f_center:f_right
       fbanks(i-1,j)= (bin_hz(i+1)-j)/(bin_hz(i+1)-bin_hz(i));  
    end
    
end
% Apply Mel-Bank onto the DFT %
Y = fbanks*dft;

% Find the Logarithm %
lg = log10(Y);

% MFCC 12 Coefficients from DCT (2-13)
mfcc1 = zeros(40,n_overlap_frames);
mfcc26 = zeros(26,n_overlap_frames);
for k=1:40
    mfcc1(k,:) = dct(lg(k,:));
   
end
mfcc26(:,:)= mfcc1(2:27,:);
% MFCC Delta
mfcc26_delta=zeros(26,n_overlap_frames);
for k=2:26
    mfcc26_delta(k-1,:)=mfcc26(k,:)-mfcc26(k-1,:);
end

% MFCC Double-Delta
mfcc26_doubledelta = zeros(26,n_overlap_frames);
for k=2:26
    mfcc26_doubledelta(k-1,:)=mfcc26_delta(k,:)-mfcc26_delta(k-1,:);
end

mfcc_sig=mfcc26;
mfcc_delta = mfcc26_delta;
mfcc_dd = mfcc26_doubledelta;
end




%%
function pitch2 = pitch(X)
fs=44100;
[~, cols] = size(X); % Overlapped Frames
n_overlap_frames =cols;
pitch1 = zeros(1,n_overlap_frames);
for j=1:cols
    %disp(j);
    auto_corr_y=xcorr(X(:,j));
    [peaks,loc] = findpeaks(auto_corr_y); 
    [max1,peak_ind]=max(peaks);
    peaks = peaks(peaks~=max(peaks));
    [max2,peak_ind2]=max(peaks);
    
    if peak_ind==1
        delta_t = loc(peak_ind);
    else
        delta_t=abs(loc(peak_ind2)-loc(peak_ind)); 
    end
    
    if isempty(delta_t)
        delta_t=0;
    end
        
    pitch1(1,j)=fs/delta_t;
end
pitch2=pitch1;
end

%%
function loud_sig = loud(X)
[~, cols] = size(X); % Overlapped Frames
Fs = 44100;
r = resample(X(:,1),160,441);
preemph = [1 -0.97];
r = filter(1,preemph,r);
r=r-mean(r); % remove DC component
s=r/max(abs(r));  %normalization
x = s;
% n_overlap_frames =cols;
%loud_f = zeros(1,n_overlap_frames);

loud_f = integratedLoudness(x,44100);

loud_sig=loud_f;
end
%%
function spec_rolloff = srolloff(X)
[~, cols] = size(X); % Overlapped Frames
n_overlap_frames =cols;
spec_r = zeros(1,n_overlap_frames);
for k=1:cols
    p = periodogram(X(:,k));
    spec_r(1,k)= prctile(p,95) ;
end
spec_rolloff=spec_r;
end
%% 
function spec_centroid = scentroid(X)
[~, cols] = size(X); % Overlapped Frames
n_overlap_frames =cols;
spec_c = zeros(1,n_overlap_frames);
for k=1:cols
        w_sum = 0; sum=0;
        [Sp,F] = periodogram(X(:,k));
        %plot(F,10*log10(Pxx));
        for j=1:length(F)
            if Sp(j)~=0
                w_sum = w_sum + 10*log10(Sp(j))*F(j);
                sum = sum + 10*log10(Sp(j));
            end
        end
        spec_c(1,k)= w_sum/sum;
end
spec_centroid=spec_c;
end
%%
function spec_zr = zerocross(X)
[~, cols] = size(X); % Overlapped Frames
n_overlap_frames =cols;
Zero_Cross = zeros(1,n_overlap_frames);
for k=1:cols
        num_zero = 0;
        for j=1:length(X(:,k))
            if (j>1)
                if(((X(j-1,k)>0) && (X(j,k)<0))||((X(j-1,k)<0) && (X(j,k)>0)))
                    num_zero= num_zero +1;
                end
            end
        end
    Zero_Cross(1,k) = num_zero;    
end
spec_zr=Zero_Cross;
end
%%
function  spec_flux = sflux(X)
[~, cols] = size(X); % Overlapped Frames
n_overlap_frames =cols;
SFlux = zeros(1,n_overlap_frames);
for k=1:cols
        if (k>1)
            sdiff = X(:,k) - X(:,k-1);
            SFlux(1,k)= sqrt(sdiff' * sdiff);
        end
end
spec_flux=SFlux;
end
%%
function spec_energy = senergy(X)
[~, cols] = size(X); % Overlapped Frames
n_overlap_frames =cols;
spec_ener = zeros(1,n_overlap_frames);
for k=1:cols
    spec_ener(1,k)= rms(X(:,k)).^2 ;
end
spec_energy=spec_ener;
end
%%
function spec_flat= spflat(X)
[~, cols] = size(X); % Overlapped Frames
n_overlap_frames =cols;
spec_flatness = zeros(1,n_overlap_frames);
for k=1:cols
    p = periodogram(X(:,k));
    
    spec_flatness(1,k)= geomean(p)/mean(p) ;
end
spec_flat=spec_flatness;
end
