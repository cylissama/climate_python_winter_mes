clear
format long

site='FARM';

%load properly formatted input data
dircli1='/Users/erappin/Documents/Mesonet/ClimateIndices/sitesTEST_CCindices/';
dircli=strcat(strcat(dircli1,site,'/'));
filethresh=strcat(dircli,strcat(site,"_CLIthresh_daily"));
load(filethresh);
fileseasonal=strcat(dircli,strcat(site,"_DATAinput_seasonal"));
load(fileseasonal);

DATAall=DATAseason.data;
DATAyear=DATAseason.year;

%DATAwinter=DATAall{1,1};
%DATAspring=DATAall{2,1};
%DATAsummer=DATAall{3,1};
%DATAfall=DATAall{4,1};
   
ny=numel(DATAseason.year);
nm=numel(DATAseason.season);
nv=numel(DATAseason.var);
nd=[90 92 92 91];

%each season is like the annual cell except the number of days is the
%number of days in the season
%Year 7 daily fall temperture (TAIR; VAR 1) DATAall{4,1}{7,1}{:,1}

for i=1:nm
    for j=1:ny
        if isempty(DATAall{i,1}{j,1})
           if j~=ny
              DATAall{i,1}{j,1}=DATAall{i,1}{j+1,1};
           else
              DATAall{i,1}{j,1}=DATAall{i,1}{j-1,1};
           end
           DATAall{i,1}{j,1}{:,:}=NaN;
           TIME=DATAall{i,1}{j,1}.TimestampCollected;
           TIME.Year=DATAyear(j);
           DATAall{i,1}{j,1}.TimestampCollected=TIME;
        end
    end
end

return

%%%%%%%%%%%%%%%%%%%%%%%CALCULATE CC INDICES%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%COLD INDICES%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%Growing Degree Days: GD4,GD10
GD4=zeros(nm,ny);
GD10=zeros(nm,ny);
for i=1:nm
    for j=1:ny
        TAIR=DATA{i,1}{1,j};
        GD4m(j,i)=sum(max((TAIR-4),0));
        GD10m(j,i)=sum(max((TAIR-10),0));
    end
end

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
for i=1:nm
    for j=1:ny
        TAIRn=DATA{i,5}{1,j};
        a1=find(TAIRn<0);
        FDm(i,j)=numel(a1);
    end
end

%Consecutive Frost Days: CFD
for i=1:nm
    for j=1:ny
        TAIRn=DATA{i,5}{1,j};
        [B,N,BI]=RunLength(TAIRn<0);
        CFDm(i,j)=max(double(B).*N,[],'omitnan');
    end
end

%Heating Degree Days: HDD
for i=1:nm
    for j=1:ny
        TAIR=DATA{i,1}{1,j};
        HDDm(i,j)=sum(18.3-TAIR,'omitnan');
    end
end

%Ice Days: ID
for i=1:nm
    for j=1:ny
        TAIRx=DATA{i,3}{1,j};
        a1=find(TAIRx<0);
        IDm(i,j)=numel(a1);
    end
end

%Cold Spell Duration Index: CSDI

%Cold Days: TG10p

%Cold Nights: TN10p

%Cold Day-Times: TX10p

%Minimum Value of Daily Maximum Temperature: TXn
for i=1:nm
    for j=1:ny
        TAIRx=DATA{i,3}{1,j};
        TXnm(i,j)=min(TAIRx,[],'omitnan');
    end
end

%Minimum Value of Daily Minimum Temperature: TNn
for i=1:nm
    for j=1:ny
        TAIRn=DATA{i,5}{1,j};
        TNnm(i,j)=min(TAIRn,[],'omitnan');
    end
end

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
for i=1:nm
    for j=1:ny
        PRCP=DATA{i,7}{1,j};
        Ra=DATA{i,14}{1,j};
        tmax=DATA{i,3}{1,j};
        tmin=DATA{i,5}{1,j};
        tmean=DATA{i,1}{1,j};
        pevapm(i,j) = mean(pet(Ra,tmax,tmin,tmean),'omitnan');
    end
end

%Standardized Precipitation - Evapotranspiration Index: SPEI

%%%%%%%%%%%%%%%%%%%%%%HEAT INDICES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Summer Days: SU 
for i=1:nm
    for j=1:ny
        TAIRx=DATA{i,3}{1,j};
        a1=find(TAIRx>25);
        SUm(i,j)=numel(a1);
    end
end

%Maximum Number of Consecultive of Summer Days: CSU
for i=1:nm
    for j=1:ny
        TAIRx=DATA{i,3}{1,j};
        [B,N,BI]=RunLength(TAIRx>25);
        CSUm(i,j)=max(double(B).*N,[],'omitnan');
    end
end

%Tropical Nights: TR
for i=1:nm
    for j=1:ny
        TAIRn=DATA{i,5}{1,j};
        a1=find(TAIRn>20);
        TRm(i,j)=numel(a1);
    end
end

%Warm Spell Duration Index: WSDI

%Warm Days: TG90p

%Warm Nights: TN90p

%Warm Day-Times: TX90p

%Maximum Value of Daily Maximum Temperature: TXx
for i=1:nm
    for j=1:ny
        TAIRx=DATA{i,3}{1,j};
        TXxm(i,j)=max(TAIRx,[],'omitnan');
    end
end

%Maximum Value of Daily Minimum Temperature: TXn
for i=1:nm
    for j=1:ny
        TAIRn=DATA{i,5}{1,j};
        TNxm(i,j)=max(TAIRn,[],'omitnan');
    end
end

%%%%%%%%%%%%%%%%%%%%%%HUMIDITY INDICES%%%%%%%%%%%%%%%%%%%%%%%%%

%Mean of Daily Relative Humidity: RH
for i=1:nm
    for j=1:ny
        RELH=DATA{i,8}{1,j};
        RHm(i,j)=mean(RELH,"omitnan");
    end
end

%%%%%%%%%%%%%%%%%%%%%%PRESSURE INDICES%%%%%%%%%%%%%%%%%%%%%%%%%

%Mean of Daily Sea Level Pressure: PP

%%%%%%%%%%%%%%%%%%%%%%%RAIN INDICES%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Precipitation Sum: RR
for i=1:nm
    for j=1:ny
        PRCP=DATA{i,7}{1,j};
        RRm(i,j)=sum(PRCP,"omitnan");
    end
end

%Wet Days: RR1
for i=1:nm
    for j=1:ny
        PRCP=DATA{i,7}{1,j};
        a1=find(PRCP>1);
        RR1m(i,j)=numel(a1);
    end
end

%Simple Daily Intensity Index: SDII
for i=1:nm
    for j=1:ny
        PRCP=DATA{i,7}{1,j};
        a1=find(PRCP>1);
        W=numel(a1);
        SDIIm(i,j)=sum(PRCP(a1),"omitnan")/W;
    end
end

%Maximum Number of Consecutive Wet Days: CWD
for i=1:nm
    for j=1:ny
        PRCP=DATA{i,7}{1,j};
        [B,N,BI]=RunLength(PRCP>25);
        CWDm(i,j)=max(double(B).*N,[],'omitnan');
    end
end

%Heavy Precipitation Days: RR10
for i=1:nm
    for j=1:ny
        PRCP=DATA{i,7}{1,j};
        a1=find(PRCP>10);
        RR10m(i,j)=numel(a1);
    end
end

%Very Heavy Precipitation Days: RR20
for i=1:nm
    for j=1:ny
        PRCP=DATA{i,7}{1,j};
        a1=find(PRCP>20);
        RR20m(i,j)=numel(a1);
    end
end

%Highest 1 Day Precitation Amount: RX1day
for i=1:nm
    for j=1:ny
        PRCP=DATA{i,7}{1,j};
        RX1daym(i,j)=max(PRCP,[],'omitnan');
    end
end

%Highest 5 Day Precitation Amount: RX5day

%Moderate Wet Days: R75p

%Precipitation Fraction Due to Moderate Wet Days: R75pTOT

%Very Wet Days: R95p

%Precipitation Fraction Due to Very Wet Days: R95pTOT

%Extremely Wet Days: R99p

%Precipitation Fraction Due to Exttremely Wet Days: R99pTOT

%%%%%%%%%%SOIL MOISTURE-PRECIPITATION INDICES%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%TEMPERATURE INDICES%%%%%%%%%%%%%%%%%%%%%

%Mean of Daily Mean Air Temperature: TG
for i=1:nm
    for j=1:ny
        TAIR=DATA{i,1}{1,j};
        TGm(i,j)=max(TAIR,[],'omitnan');
    end
end

%Mean of Daily Mean Dewpoint Temperature: TdG
for i=1:nm
    for j=1:ny
        DWPT=DATA{i,2}{1,j};
        TdGm(i,j)=mean(DWPT,"omitnan");
    end
end

%Mean of Daily Minimum Air Temperature: TN
for i=1:nm
    for j=1:ny
        TAIRn=DATA{i,5}{1,j};
        TNm(i,j)=mean(TAIRn,"omitnan");
    end
end

%Mean of Daily Minimum Dewpoint Temperature: TdN
for i=1:nm
    for j=1:ny
        DWPTn=DATA{i,6}{1,j};
        TdNm(i,j)=mean(DWPTn,"omitnan");
    end
end

%Mean of Daily Maximum Air Temperature: TX
for i=1:nm
    for j=1:ny
        TAIRx=DATA{i,3}{1,j};
        TXm(i,j)=mean(TAIRx,"omitnan");
    end
end

%Mean of Daily Maximum Dewpoint Temperature: TdX
for i=1:nm
    for j=1:ny
        DWPTx=DATA{i,4}{1,j};
        TdXm(i,j)=mean(DWPTx,"omitnan");
    end
end

%Mean of Diurnal Air Temperature Range: DTR
for i=1:nm
    for j=1:ny
        TAIRn=DATA{i,5}{1,j};
        TAIRx=DATA{i,3}{1,j};
        DTRm(i,j)=sum(TAIRx-TAIRn,"omitnan")/numel(TAIRx);
    end
end

%Mean of Diurnal Dewpoint Temperature Range: DTdR
for i=1:nm
    for j=1:ny
        DWPTn=DATA{i,6}{1,j};
        DWPTx=DATA{i,4}{1,j};
        DTdRm(i,j)=sum(DWPTx-DWPTn,"omitnan")/numel(DWPTx);
    end
end

%Intra-Period Extreme Temperature Range: ETR
for i=1:nm
    for j=1:ny
        TAIRn=DATA{i,5}{1,j};
        TAIRx=DATA{i,3}{1,j};
        ETRm(i,j)=max( ((TAIRx)-min(TAIRn)),[],'omitnan' );
    end
end

%Mean Absolute Day to Day Difference in DTR: vDTR 

%%%%%%%%%%%%%%%%%%%%%%%%WIND INDICES%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Maximum Daily Wind Dpeed Gust: FXx
for i=1:nm
    for j=1:ny
        WSMX=DATA{i,13}{1,j};
        FXxm(i,j)=convvel(max(WSMX,[],"omitnan"),'m/s','mph');
    end
end

%Days With Daily Average Wind Speed Exceeding 10.8 m/s: FG11

%Calm Days: FGcalm

%Mean od Daily Mean Wind Strength: FG
for i=1:nm
    for j=1:ny
        WSPD=DATA{i,12}{1,j};
        FGm(i,j)=sum(WSPD,"omitnan")/numel(WSPD);
    end
end

%Wind Direction
for i=1:nm
    for j=1:ny
        WDIR=DATA{i,11}{1,j};
        %Days With Southerly Winds: DDsouth
        a1=find(WDIR<=225 & WDIR>135);
        DDsouthm(i,j)=numel(a1);
        %Days With Easterly Winds: DDeast
        b1=find(WDIR<=135 & WDIR>45);
        DDeastm(i,j)=numel(b1);
        %Days With Westerly Winds: DDwest
        c1=find(WDIR<=315 & WDIR>225);
        DDwestm(i,j)=numel(c1);
        %Days With Northerly Winds: DDnorth
        d1=find(WDIR<=45 | WDIR>315);
        DDnorthm(i,j)=numel(d1);
    end
end