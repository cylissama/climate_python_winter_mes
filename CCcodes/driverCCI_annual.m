
site='BMTN';
%load properly formatted input data
dircli1='/Volumes/Mesonet/winter_break/CCdata/';
dircli=strcat(strcat(dircli1,site,'/'));
filethresh=strcat(dircli,strcat(site,"_CLIthresh_daily"));
load(filethresh);

% fileannual=strcat(dircli,strcat(site,"_DATAinput_annual"));
% load(fileannual);

ny=numel(DATAannual.year);
nv=numel(DATAannual.var);
nd=365;

DATAyear=DATAannual.year;
DATAall=DATAannual.data;
%DATAall{ny,1}(nd,nv)
%Year 1 daily temperture DATA{1,1}{:,1}

%%%%%%%%%%%%%%%%%%%%%%%CALCULATE CC INDICES%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%COLD INDICES%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%Growing Degree Days: GD4,GD10
GD4=zeros(ny,1);
GD10=zeros(ny,1);
for j=1:ny
    TAIR=DATAall{j,1}{:,1};
    GD4m(j,1)=sum(max((TAIR-4),0));
    GD10m(j,1)=sum(max((TAIR-10),0));
end
hfig=figure;
xaxis=1:1:numel(GD4m);
mdlr=fitlm(xaxis,GD4m,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(GD4m,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
%title(['ISFS 2-meter Temperature ({\circ}C)' ' '   'July 2018'] ,'fontsize',14);
title(strcat(site,' GD4 - Growing degree days (sum of TAIRg > 4 ◦C) (◦C)'),'fontsize',14);
xticks(1:1:numel(GD4m));
xlim([1 numel(GD4m)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('({\circ}C)','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_GD4'));
print(hfig,fileout,'-dpng','-r600');

%%%%%Growing Season Length (GSL)
%thresh=5;
%for i=1:1
%    TAIR=DATA{1,1}{1,i};
%    TairCalc=TAIR(IndicesY(i):(IndicesY(i+1)-IndicesY(i)));
%    TimeCalc=TIME(IndicesY(i):(IndicesY(i+1)-IndicesY(i)));
%    %start
%    [B,N,BI]=RunLength(TairCalc > thresh);
%    ix = find(N>thresh,1);
%    t1=TIME(ix+1);
%    %end
%    summerT=datetime(YEAR(i),07,01);
%    IndicesT=datefind(summerT,TimeCalc)-IndicesY(i)+1;
%    TIME2=TimeCalc(IndicesT:numel(TairCalc));
%    TAIR2=TairCalc(IndicesT:numel(TairCalc));
%    [B2,N2,BI2]=RunLength(TAIR2 < thresh);
%    %%%%%%%%%%%%%%%%%%%NEED TO FIX%%%%%%%%%%%%%%%%%
%    %if B2==1 & N2>thresh
%    %   ix2
%    %   %t2=TIME2(ix2+1);
%    %end
%    %GSL(i)=duration({t1;t2}); 
%end

%Frost Days: FD
for j=1:ny
    TAIRn=DATAall{j,1}{:,3};
    a1=find(TAIRn<0);
    FDm(j)=numel(a1);
end
hfig=figure;
xaxis=1:1:numel(FDm);
mdlr=fitlm(xaxis,FDm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(FDm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' FD - Frost Days (TAIRn < 0 ◦C) (Days)'),'fontsize',14);
xticks(1:1:numel(FDm));
xlim([1 numel(FDm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('(Days)','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_FD'));
print(hfig,fileout,'-dpng','-r600');

%Consecutive Frost Days: CFD
for j=1:ny
    TAIRn=DATAall{j,1}{:,3};
    [B,N,BI]=RunLength(TAIRn<0);
    CFDm(j)=max(double(B).*N,[],'omitnan');
end
hfig=figure;
xaxis=1:1:numel(CFDm);
mdlr=fitlm(xaxis,CFDm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(CFDm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' CFD - Maximum # of Consecutive Frost Days (TAIRn < 0 ◦C) (Days)'),'fontsize',14);
xticks(1:1:numel(CFDm));
xlim([1 numel(CFDm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('(Days)','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_CFD'));
print(hfig,fileout,'-dpng','-r600');

%Heating Degree Days: HDD
for j=1:ny
    TAIR=DATAall{j,1}{:,1};
    HDDm(j)=sum(18.333-TAIR,[],'omitnan');
end
hfig=figure;
xaxis=1:1:numel(HDDm);
mdlr=fitlm(xaxis,HDDm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(HDDm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
%title(['ISFS 2-meter Temperature ({\circ}C)' ' '   'July 2018'] ,'fontsize',14);
title(strcat(site,' HDD - Heating degree days (sum of 18.333◦C - TAIRg) (◦C)'),'fontsize',14);
xticks(1:1:numel(HDDm));
xlim([1 numel(HDDm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('({\circ}C)','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_HDD'));
print(hfig,fileout,'-dpng','-r600');

%Ice Days: ID
for j=1:ny
        TAIRx=DATAall{j,1}{:,2};
        a1=find(TAIRx<0);
        IDm(j)=numel(a1);
end
hfig=figure;
xaxis=1:1:numel(IDm);
mdlr=fitlm(xaxis,IDm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(IDm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' ID - Ice Days (TAIRx < 0 ◦C) (Days)'),'fontsize',14);
xticks(1:1:numel(IDm));
xlim([1 numel(IDm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('(Days)','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_ID'));
print(hfig,fileout,'-dpng','-r600');

%Cold Spell Duration Index: CSDI

%Cold Days: TG10p

%Cold Nights: TN10p

%Cold Day-Times: TX10p

%Minimum Value of Daily Maximum Temperature: TXn
for j=1:ny
    TAIRx=DATAall{j,1}{:,2};
    TXnm(j)=min(TAIRx,[],'omitnan');
end
hfig=figure;
xaxis=1:1:numel(TXnm);
mdlr=fitlm(xaxis,TXnm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(TXnm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' TXn Minimum Value of Daily Maximum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(TXnm));
xlim([1 numel(TXnm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('({\circ}C)','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_TXn'));
print(hfig,fileout,'-dpng','-r600');


%Minimum Value of Daily Minimum Temperature: TNn
for j=1:ny
    TAIRn=DATAall{j,1}{:,3};
    TNnm(j)=min(TAIRn,[],'omitnan');
end
hfig=figure;
xaxis=1:1:numel(TNnm);
mdlr=fitlm(xaxis,TNnm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(TNnm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' TXn Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(TNnm));
xlim([1 numel(TNnm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('({\circ}C)','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_TNn'));
print(hfig,fileout,'-dpng','-r600');

%%%%%%%%%%%%%%%%%%%%%%COMPUND INDICES%%%%%%%%%%%%%%%%%%%%%%%%%%

%Cold Dry Days: CD

%Cold Wet Days: CW

%Warm Dry Days: WD

%Warm Wet Days: WW

%%%%%%%%%%%%%%%%%%%%%%DROUGHT INDICES%%%%%%%%%%%%%%%%%%%%%%%%%%

%Maximum Number of Consecutive Dry Days: CDD

%Three Month Standarized Precipitation Index: SPI3

%Six Month Standarized Precipitation Index: SPI6

%Potential Evapotranspiration: PET
for j=1:ny
    PRCP=DATAall{j,1}{:,5};
    Ra=DATAall{j,1}{:,9};
    tmax=DATAall{j,1}{:,2};
    tmin=DATAall{j,1}{:,3};
    tmean=DATAall{j,1}{:,1};

    % Compute PET using the custom function
    PET_values = pet(Ra, tmax, tmin, tmean);
    pevapm(j) = mean(PET_values, 'omitnan') .* 25.4; % Convert to mm if needed

    %pevapm(j) = mean(pet(Ra,tmax,tmin,tmean),'omitnan').*25.4;
end
hfig=figure;
xaxis=1:1:numel(pevapm);
mdlr=fitlm(xaxis,pevapm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(pevapm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' PET - Potential Evapotranspiration (mm)'),'fontsize',14);
xticks(1:1:numel(pevapm));
xlim([1 numel(pevapm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('(mm)','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_PET'));
print(hfig,fileout,'-dpng','-r600');

%Standardized Precipitation - Evapotranspiration Index: SPEI
%return
%%%%%%%%%%%%%%%%%%%%%%HEAT INDICES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Summer Days: SU 
for j=1:ny
        TAIRx=DATAall{j,1}{:,2};
        a1=find(TAIRx>25);
        SUm(j)=numel(a1);
end
hfig=figure;
xaxis=1:1:numel(SUm);
mdlr=fitlm(xaxis,SUm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(SUm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' SU - Summer Days (TAIRx > 25 ◦C) (Days)'),'fontsize',14);
%title(strcat(site,' SU Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(SUm));
xlim([1 numel(SUm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('Days','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_SU'));
print(hfig,fileout,'-dpng','-r600');

%Maximum Number of Consecultive of Summer Days: CSU
for j=1:ny
        TAIRx=DATAall{j,1}{:,2};
        [B,N,BI]=RunLength(TAIRx>25);
        CSUm(j)=max(double(B).*N,[],'omitnan');
end
hfig=figure;
xaxis=1:1:numel(CSUm);
mdlr=fitlm(xaxis,CSUm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(CSUm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' CSU - Maximum # of Consecutive Summer Days (TAIRx > 25 ◦C) (Days)'),'fontsize',14);
xticks(1:1:numel(CSUm));
xlim([1 numel(CSUm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('Days','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_CSU'));
print(hfig,fileout,'-dpng','-r600');

%Tropical Nights: TR
for j=1:ny
    TAIRn=DATAall{j,1}{:,3};
    a1=find(TAIRn>20);
    TRm(j)=numel(a1);
end
hfig=figure;
xaxis=1:1:numel(TRm);
mdlr=fitlm(xaxis,TRm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(TRm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' TR - Tropical Nights (TAIRn > 20 ◦C) (Days)'),'fontsize',14);
xticks(1:1:numel(TRm));
xlim([1 numel(TRm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('Days','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_TR'));
print(hfig,fileout,'-dpng','-r600');

%Warm Spell Duration Index: WSDI

%Warm Days: TG90p

%Warm Nights: TN90p

%Warm Day-Times: TX90p

%Maximum Value of Daily Maximum Temperature: TXx
for j=1:ny
    TAIRx=DATAall{j,1}{:,2};
    TXxm(j)=max(TAIRx,[],'omitnan');
end
hfig=figure;
xaxis=1:1:numel(TXxm);
mdlr=fitlm(xaxis,TXxm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(TXxm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' TXx Maximum Value of Daily Maximum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(TXxm));
xlim([1 numel(TXxm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('({\circ}C)','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_TXx'));
print(hfig,fileout,'-dpng','-r600');

%Maximum Value of Daily Minimum Temperature: TNx
for j=1:ny
    TAIRn=DATAall{j,1}{:,3};
    TNxm(j)=max(TAIRn,[],'omitnan');
end
hfig=figure;
xaxis=1:1:numel(TNxm);
mdlr=fitlm(xaxis,TNxm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(TNxm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' TNx Maximum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(TXnm));
xlim([1 numel(TXnm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('({\circ}C)','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_TXn'));
print(hfig,fileout,'-dpng','-r600');
%%%%%%%%%%%%%%%%%%%%%%HUMIDITY INDICES%%%%%%%%%%%%%%%%%%%%%%%%%

%Mean of Daily Relative Humidity: RH
for j=1:ny
    RELH=DATAall{j,1}{:,7};
    RHm(j)=mean(RELH,"omitnan");
end
hfig=figure;
xaxis=1:1:numel(RHm);
mdlr=fitlm(xaxis,RHm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(RHm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' RH - Mean Relative Humidity (%)'),'fontsize',14);
%title(strcat(site,' SU Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(RHm));
xlim([1 numel(RHm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('%','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_RH'));
print(hfig,fileout,'-dpng','-r600');

%Mean of Daily Dew Point Temperature: DP
for j=1:ny
    DWPT=DATAall{j,1}{:,4};
    DPm(j)=mean(DWPT,"omitnan");
end
hfig=figure;
xaxis=1:1:numel(DPm);
mdlr=fitlm(xaxis,DPm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(DPm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' DP - Mean Dew Point Temperature ({\circ}C)'),'fontsize',14);
%title(strcat(site,' SU Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(RHm));
xlim([1 numel(RHm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('{\circ}C','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_DP'));
print(hfig,fileout,'-dpng','-r600');

%%%%%%%%%%%%%%%%%%%%%%PRESSURE INDICES%%%%%%%%%%%%%%%%%%%%%%%%%

%Mean of Daily Sea Level Pressure: PP

%%%%%%%%%%%%%%%%%%%%%%%RAIN INDICES%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Precipitation Sum: RR
for j=1:ny
    PRCP=DATAall{j,1}{:,5};
    RRm(j)=sum(PRCP,"omitnan");
end
hfig=figure;
xaxis=1:1:numel(RRm);
mdlr=fitlm(xaxis,RRm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(RRm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' RR - Precipitation Sum (mm)'),'fontsize',14);
%title(strcat(site,' SU Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(RRm));
xlim([1 numel(RRm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('mm','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_RR'));
print(hfig,fileout,'-dpng','-r600');

%Wet Days: RR1
for j=1:ny
    PRCP=DATAall{j,1}{:,5};
    a1=find(PRCP>1);
    RR1m(j)=numel(a1);
end
hfig=figure;
xaxis=1:1:numel(RR1m);
mdlr=fitlm(xaxis,RR1m,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(RR1m,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' RR1 - Wet Days (RR >= 1mm) (Days)'),'fontsize',14);
%title(strcat(site,' SU Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(RR1m));
xlim([1 numel(RR1m)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('Days','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_RR1'));
print(hfig,fileout,'-dpng','-r600');

%Simple Daily Intensity Index: SDII
for j=1:ny
        PRCP=DATAall{j,1}{:,5};
        a1=find(PRCP>1);
        W=numel(a1);
        SDIIm(j)=sum(PRCP(a1),"omitnan")/W;
end
hfig=figure;
xaxis=1:1:numel(SDIIm);
mdlr=fitlm(xaxis,SDIIm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(SDIIm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' SDII - Simple Daily Intensity Index (mm/wetday)','fontsize',14));
%title(strcat(site,' SU Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(SDIIm));
xlim([1 numel(SDIIm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('mm/wetday','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_SDII'));
print(hfig,fileout,'-dpng','-r600');

%Maximum Number of Consecutive Wet Days: CWD
for j=1:ny
        PRCP=DATAall{j,1}{:,5};
        [B,N,BI]=RunLength(PRCP>1);
        CWDm(j)=max(double(B).*N,[],'omitnan');
end
hfig=figure;
xaxis=1:1:numel(CWDm);
mdlr=fitlm(xaxis,CWDm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(CWDm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' CWD - Maximum # of Consecutive Wet Days (RR > 1mm) (Days)'),'fontsize',14);
xticks(1:1:numel(CWDm));
xlim([1 numel(CWDm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('Days','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_CWD'));
print(hfig,fileout,'-dpng','-r600');

%Heavy Precipitation Days: RR10
for j=1:ny
        PRCP=DATAall{j,1}{:,5};
        a1=find(PRCP>10);
        RR10m(j)=numel(a1);
end
hfig=figure;
xaxis=1:1:numel(RR10m);
mdlr=fitlm(xaxis,RR10m,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(RR10m,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' RR10 - Heavy Precipitation Days (RR >= 10mm) (Days)'),'fontsize',14);
%title(strcat(site,' SU Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(RR10m));
xlim([1 numel(RR10m)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('Days','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_RR10'));
print(hfig,fileout,'-dpng','-r600');

%Very Heavy Precipitation Days: RR20
for j=1:ny
        PRCP=DATAall{j,1}{:,5};
        a1=find(PRCP>20);
        RR20m(j)=numel(a1);
end
hfig=figure;
xaxis=1:1:numel(RR20m);
mdlr=fitlm(xaxis,RR20m,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(RR20m,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' RR20 - Very Heavy Precipitation Days (RR >= 20mm) (Days)'),'fontsize',14);
%title(strcat(site,' SU Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(RR20m));
xlim([1 numel(RR20m)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('Days','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_RR20'));
print(hfig,fileout,'-dpng','-r600');

%Highest 1 Day Precitation Amount: RX1day
for j=1:ny
    PRCP=DATAall{j,1}{:,5};
    RX1daym(j)=max(PRCP,[],'omitnan');
end
hfig=figure;
xaxis=1:1:numel(RX1daym);
mdlr=fitlm(xaxis,RX1daym,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(RX1daym,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' RX1day - Highest 1 Day Precipitation (mm)'),'fontsize',14);
%title(strcat(site,' SU Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(RX1daym));
xlim([1 numel(RX1daym)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('mm','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_RX1day'));
print(hfig,fileout,'-dpng','-r600');

%Highest 5 Day Precitation Amount: RX5day
for j=1:ny
    PRCP=DATAall{j,1}{:,5};
    RX5day=movsum(PRCP,5);
    RX5daym(j)=max(RX5day);
end
hfig=figure;
xaxis=1:1:numel(RX5daym);
mdlr=fitlm(xaxis,RX5daym,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(RX5daym,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' RX5day - Highest 5 Day Precipitation (mm)'),'fontsize',14);
%title(strcat(site,' SU Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(RX5daym));
xlim([1 numel(RX5daym)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('Days','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_RX5day'));
print(hfig,fileout,'-dpng','-r600');

%Moderate Wet Days: R75p

%Precipitation Fraction Due to Moderate Wet Days: R75pTOT

%Very Wet Days: R95p

%Precipitation Fraction Due to Very Wet Days: R95pTOT

%Extremely Wet Days: R99p

%Precipitation Fraction Due to Exttremely Wet Days: R99pTOT

%%%%%%%%%%SOIL MOISTURE-PRECIPITATION INDICES%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%TEMPERATURE INDICES%%%%%%%%%%%%%%%%%%%%%

%Mean of Daily Mean Air Temperature: TG
for j=1:ny
    TAIR=DATAall{j,1}{:,1};
    TGm(j)=mean(TAIR,'omitnan');
end
hfig=figure;
xaxis=1:1:numel(TGm);
mdlr=fitlm(xaxis,TGm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(TGm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' TG - Mean Temperature ({\circ}C)'),'fontsize',14);
%title(strcat(site,' SU Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(TGm));
xlim([1 numel(TGm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('{\circ}C','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_TG'));
print(hfig,fileout,'-dpng','-r600');

%Mean of Daily Minimum Air Temperature: TN
for j=1:ny
        TAIRn=DATAall{j,1}{:,3};
        TNm(j)=mean(TAIRn,"omitnan");
end
hfig=figure;
xaxis=1:1:numel(TNm);
mdlr=fitlm(xaxis,TNm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(TNm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' TN - Mean Minimum Temperature ({\circ}C)'),'fontsize',14);
%title(strcat(site,' SU Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(TNm));
xlim([1 numel(TNm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('{\circ}C','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_TN'));
print(hfig,fileout,'-dpng','-r600');

%Mean of Daily Maximum Air Temperature: TX
for j=1:ny
    TAIRx=DATAall{j,1}{:,2};
    TXm(j)=mean(TAIRx,"omitnan");
end
hfig=figure;
xaxis=1:1:numel(TXm);
mdlr=fitlm(xaxis,TXm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(TXm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' TX - Mean Maximum Temperature ({\circ}C)'),'fontsize',14);
%title(strcat(site,' SU Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(TXm));
xlim([1 numel(TXm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('{\circ}C','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_TX'));
print(hfig,fileout,'-dpng','-r600');

%Mean of Diurnal Air Temperature Range: DTR
for j=1:ny
    TAIRn=DATAall{j,1}{:,3};
    TAIRx=DATAall{j,1}{:,2};
    DTRm(j)=sum(TAIRx-TAIRn,"omitnan")/numel(TAIRx);
end
hfig=figure;
xaxis=1:1:numel(DTRm);
mdlr=fitlm(xaxis,DTRm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(DTRm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' DTR - Mean Diurnal Temperature Range ({\circ}C)'),'fontsize',14);
%title(strcat(site,' SU Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(DTRm));
xlim([1 numel(DTRm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('{\circ}C','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_DTR'));
print(hfig,fileout,'-dpng','-r600');

%Intra-Period Extreme Temperature Range: ETR
for j=1:ny
    TAIRn=DATAall{j,1}{:,3};
    TAIRx=DATAall{j,1}{:,2};
    ETRm(j)=max( ((TAIRx)-min(TAIRn)),[],'omitnan' );
end
hfig=figure;
xaxis=1:1:numel(ETRm);
mdlr=fitlm(xaxis,ETRm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(ETRm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' ETR - Mean Intra-Period Extreme Temperature Range ({\circ}C)'),'fontsize',14);
%title(strcat(site,' SU Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(ETRm));
xlim([1 numel(ETRm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('{\circ}C','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_ETR'));
print(hfig,fileout,'-dpng','-r600');

%Mean Absolute Day to Day Difference in DTR: vDTR 

%%%%%%%%%%%%%%%%%%%%%%%%WIND INDICES%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Maximum Daily Wind Dpeed Gust: FXx
for j=1:ny
    WSMX=DATAall{j,1}{:,12};
    FXxm(j)=convvel(max(WSMX,[],"omitnan"),'m/s','mph');
end
hfig=figure;
xaxis=1:1:numel(FXxm);
mdlr=fitlm(xaxis,FXxm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(FXxm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' FXx - Maximum Daily Wind Speed Gust (ms^{-1}'),'fontsize',14);
%title(strcat(site,' SU Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(FXxm));
xlim([1 numel(FXxm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('ms^{-1}','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_FXx'));
print(hfig,fileout,'-dpng','-r600');

%Days With Daily Average Wind Speed Exceeding 10.8 m/s: FG11

%Calm Days: FGcalm

%Mean of Daily Mean Wind Strength: FG
for j=1:ny
        WSPD=DATAall{j,1}{:,11};
        FGm(j)=sum(WSPD,"omitnan")/numel(WSPD);
end
hfig=figure;
xaxis=1:1:numel(FGm);
mdlr=fitlm(xaxis,FGm,'RobustOpts','on');
Tau=mdlr.Coefficients.Estimate;
plot(FGm,'b','LineWidth',2)
hold on
plot(Tau(1)+Tau(2)*xaxis,'b--','LineWidth',2)
grid on
title(strcat(site,' FG - Mean Daily Wind Speed (ms^{-1}'),'fontsize',14);
%title(strcat(site,' SU Minimum Value of Daily Minimum Temperature - ({\circ}C)'),'fontsize',14);
xticks(1:1:numel(FGm));
xlim([1 numel(FGm)]);
xticklabels(string(DATAyear));
xlabel('YEAR','fontsize',14);
%ax.YLim=([3500 4750]);
%yticks([3500 3750 4000 4250 4500 4750]);
ylabel('ms^{-1}','fontsize',14);
set(gca,'fontsize',14);
legend('Observed','Trend','FontSize',14,'Orientation','vertical','Location','Best')
fileout=strcat('/Volumes/Mesonet/winter_break/output_data/figures/',strcat(site,'_FG'));
print(hfig,fileout,'-dpng','-r600');

%Wind Direction
for j=1:ny
        WDIR=DATAall{j,1}{:,10};
        %Days With Southerly Winds: DDsouth
        a1=find(WDIR<=225 & WDIR>135);
        DDsouthm(j)=numel(a1);
        %Days With Easterly Winds: DDeast
        b1=find(WDIR<=135 & WDIR>45);
        DDeastm(j)=numel(b1);
        %Days With Westerly Winds: DDwest
        c1=find(WDIR<=315 & WDIR>225);
        DDwestm(j)=numel(c1);
        %Days With Northerly Winds: DDnorth
        d1=find(WDIR<=45 | WDIR>315);
        DDnorthm(j)=numel(d1);
end

disp("FINISHED")