%数据读入及初始化处理
load('ECGData.mat');  %Loading ECG database
data = ECGData.Data;  %Getting Database
labels = ECGData.Labels;  %Getting Labels

ARR = data(1:96,:);  %采用前96个数据
CHF = data(97:126,:);
NSR = data(127:162,:);

signallength = 512;
%Defining filters for CWT with amor wavelet and 12 filters per obtave
fb = cwtfilterbank('SignalLength', signallength, 'Wavelet', 'amor', 'VoicesPerOctave', 12);

%Making folders
mkdir('ecgdataset2');  %Main folder
mkdir('ecgdataset2\arr');  %Sub folder
mkdir('ecgdataset2\chf');
mkdir('ecgdataset2\nsr');

ecgtype = {'ARR', 'CHF', 'NSR'};

%Function to convert ECGto Image
ecg2cwtscg(ARR, fb, ecgtype{1});
ecg2cwtscg(CHF, fb, ecgtype{2});
ecg2cwtscg(NSR, fb, ecgtype{3});