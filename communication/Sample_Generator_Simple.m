%%
% Sample_Generator_Simple.m
%
% Description: Generate a dataset of noisy IQ samples for a given
% modulation scheme and interference constellation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all

%% General Paramters
VISUALIZE = 1; % Set to 1 to show received constellations

%% Simulation Parameters
Len_block = 50; % Number of samples per block
Num_blocks = 1000;  % Number of blocks observed in dataset

SNR = 60;               % Signal to noise ratio
INR = 5;                % Interference to noise ratio
SINR = SNR/(INR + 1);   % Signal to Interference plus Noise Ratio

p_int = 0.8; %Probability of interference being present in a given block

% QAM Constellations (normalized to average symbol power of 1)
N_Constellations = 2; % Considering 4QAM and 16QAM
Constellation_4  = sqrt(0.5)*[1+1i; 1-1i; -1+1i; -1-1i];
Constellation_16 = [-3+3i; -1+3i; 1+3i; 3+3i; ...
                    -3+1i; -1+1i; 1+1i; 3+1i; ...
                    -3-1i; -1-1i; 1-1i; 3-1i; ...
                    -3-3i; -1-3i; 1-3i; 3-3i];
Constellation_16 = Constellation_16./sqrt(mean(abs(Constellation_16).^2));



%% Define relative powers;
P_n = 1;        % Normalize to noise power of 1
P_x = SNR*P_n;  % Signal Power (received)
P_i = INR*P_n;  % Interference Power (received)

%% Allocate Array
L_S_x = zeros(Len_block,Num_blocks);
L_S_i = zeros(Len_block,Num_blocks);
X   = zeros(Len_block,Num_blocks);
I   = zeros(Len_block,Num_blocks);


%% Generate signal and interference Symbols
L_Constellations = randi(N_Constellations,1,Num_blocks);
L_Interference   = rand(1,Num_blocks) < p_int; % Interference either present or not
for block = 1:Num_blocks

    % Define signal constellation for the block
    Constellation = L_Constellations(block);
    switch Constellation
        case 1
            Constellation_S = Constellation_4;
        case 2
            Constellation_S = Constellation_16;
        otherwise
            Constellation_S = Constellation_4;
    end
    
    % Define interference constellation for the block
    Constellation_I = L_Interference(block)*Constellation_4;

    % Randomly select symbols for the given block
    L_S_x(:,block) = randi(length(Constellation_S),Len_block,1);
    L_S_i(:,block) = randi(length(Constellation_I),Len_block,1);

    X(:,block) = Constellation_S(L_S_x(:,block));
    I(:,block) = Constellation_I(L_S_i(:,block));
end
N = sqrt(0.5)*P_n*(randn(Len_block,Num_blocks) + 1i*randn(Len_block,Num_blocks));


%% Generate received signal
Y = P_x*X + P_i*I + N;



save data1.mat L_Constellations L_Interference L_S_x L_S_i
save data1_label.mat Y