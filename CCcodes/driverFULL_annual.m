clear
format long

site='BMTN';

fileMES1='/BMTN/01-Mar-2014_01-Aug-2023_BMTN_daily.mat';
%/Volumes/Mesonet/winter_break/CCdata/BMTN/01-Mar-2014_01-Aug-2023_BMTN_daily.mat
%load Mesonet, Climate Data and Thresholds
dircli1='/Volumes/Mesonet/winter_break/CCdata/';
dircli=strcat(strcat(dircli1,site,'/'));
filethresh=strcat(dircli,strcat(site,"_CLIthresh_daily.mat"));
load(filethresh);
filemes=strcat(dircli1,fileMES1);
load(filemes);

TIME_full=TT_dailyMES.TimestampCollected;
sTIME_full=TIME_full(1);
eTIME_full=TIME_full(end);
YEAR_full=unique(TIME_full.Year);
[sYf,sMf,sDf]=ymd(TIME_full(1));
[eYf,eMf,eDf]=ymd(TIME_full(end));
dateS=datetime(sYf+1,01,01);
if dateS >= sTIME_full
   isD=find(all(ismember(TIME_full,dateS),2));
end
dateE=datetime(eYf-1,12,31); 
if dateE <= eTIME_full
   ieD=find(all(ismember(TIME_`full,dateE),2));
end

TT=TT_dailyMES(isD:ieD,:);
TIME=TT.TimestampCollected;

VAR=["TAIR_month" "DWPT_month" "TAIRx_month" "TAIRn_month" "PRCP_month" "RELH_month" ...
     "PRES_month" "SM02_month" "WDIR_month" "WSPD_month" "WSMX_month" "SRAD_month"]';
YEAR=unique(TIME.Year);

myStructA.var=VAR;
myStructA.year=YEAR;

ny=numel(YEAR);
nv=numel(VAR);

Md=cell(ny,nv);
My=cell(ny,nv);

yoD = year(TIME);         
moD = month(TIME);   

for i=1:ny
    k=2009+i;
    iy = ismember(yoD,k);
    TIME_annual=TIME(iy,:);
    for j=1:nv
        TTnew=getData_annual(TT,iy,TIME);
        TTcal=TIME;
        TTyears = year(TTcal);
        [~,TTindYears] = findgroups(TTyears);
        Vartt=TTnew{:,j};
        TTy{:,i}=TTnew;
        Ad{i,j}=TTy;
        Ay{i,j}=TTindYears;
    end
end

myStructA.year=Ay;
myStructA.data=Ad;

DATA=myStructA.data;

IndicesY=ones(ny,1);
for i=1:ny
    yearT(i)=datetime(double(YEAR(i)),01,01);
    IndicesY(i)=datefind(yearT(i),TIME_full);
end

%%%%%%%%%%%%%%%%%%function for annual data%%%%%%%%%%%%%%%%%%%%%
function [TT_annual]=getData_annual(TT,iy,TIME)

TIME_annual=TIME(iy);

TAIR_A=TT.TAIR;
TAIR_annual=TAIR_A(iy,:);
TT_TAIR_an=timetable(TIME_annual,TAIR_annual);

DWPT_A=TT.DWPT;
DWPT_annual=DWPT_A(iy,:);
TT_DWPT_an=timetable(TIME_annual,DWPT_annual);

TAIRx_A=TT.TAIRx;
TAIRx_annual=TAIRx_A(iy,:);
TTx_TAIR_an=timetable(TIME_annual,TAIRx_annual);

TAIRn_A=TT.TAIRn;
TAIRn_annual=TAIRn_A(iy,:);
TTn_TAIR_an=timetable(TIME_annual,TAIRn_annual);

PRCP_A=TT.PRCP;
PRCP_annual=PRCP_A(iy,:);
TT_PRCP_an=timetable(TIME_annual,PRCP_annual);

RELH_A=TT.RELH;
RELH_annual=RELH_A(iy,:);
TT_RELH_an=timetable(TIME_annual,RELH_annual);

PRES_A=TT.PRES;
PRES_annual=PRES_A(iy,:);
TT_PRES_an=timetable(TIME_annual,PRES_annual);

SM02_A=TT.SM02;
SM02_annual=SM02_A(iy,:);
TT_SM02_an=timetable(TIME_annual,SM02_annual);

WDIR_A=TT.WDIR;
WDIR_annual=WDIR_A(iy,:);
TT_WDIR_an=timetable(TIME_annual,WDIR_annual);

WSPD_A=TT.WSPD;
WSPD_annual=WSPD_A(iy,:);
TT_WSPD_an=timetable(TIME_annual,WSPD_annual);

WSMX_A=TT.WSMX;
WSMX_annual=WSMX_A(iy,:);
TT_WSMX_an=timetable(TIME_annual,WSMX_annual);

SRAD_A=TT.SRAD;
SRAD_annual=SRAD_A(iy,:);
TT_SRAD_an=timetable(TIME_annual,SRAD_annual);

TT_annual=[TT_TAIR_an TT_DWPT_an TTx_TAIR_an TTn_TAIR_an ...
    TT_PRCP_an TT_RELH_an TT_PRES_an TT_SM02_an TT_WDIR_an TT_WSPD_an TT_WSMX_an TT_SRAD_an];

end

%%%%%%%%%%%%%%%%%%%%%%%%INDEX CALCULATION%%%%%%%%%%%%%%%%%%%%
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




%%%%%%%%%%%%%%%%%%function for monthly data%%%%%%%%%%%%%%%%%%%%%
function [TT_month]=getData_month(TT,mo,TIME)

TIME_month=TIME(mo,:);

TAIR_M=TT.TAIR;
TAIR_month=TAIR_M(mo,:);
TT_TAIR_mo=timetable(TIME_month,TAIR_month);

DWPT_M=TT.DWPT;
DWPT_month=DWPT_M(mo,:);
TT_DWPT_mo=timetable(TIME_month,DWPT_month);

TAIRx_M=TT.TAIRx;
TAIRx_month=TAIRx_M(mo,:);
TT_TAIRx_mo=timetable(TIME_month,TAIRx_month);

TAIRn_M=TT.TAIRn;
TAIRn_month=TAIR_M(mo,:);
TT_TAIRn_mo=timetable(TIME_month,TAIRn_month);

PRCP_M=TT.PRCP;
PRCP_month=PRCP_M(mo,:);
TT_PRCP_mo=timetable(TIME_month,PRCP_month);

RELH_M=TT.RELH;
RELH_month=RELH_M(mo,:);
TT_RELH_mo=timetable(TIME_month,RELH_month);

PRES_M=TT.PRES;
PRES_month=PRES_M(mo,:);
TT_PRES_mo=timetable(TIME_month,PRES_month);

SM02_M=TT.SM02;
SM02_month=SM02_M(mo,:);
TT_SM02_mo=timetable(TIME_month,SM02_month);

WDIR_M=TT.WDIR;
WDIR_month=WDIR_M(mo,:);
TT_WDIR_mo=timetable(TIME_month,WDIR_month);

WSPD_M=TT.WSPD;
WSPD_month=WSPD_M(mo,:);
TT_WSPD_mo=timetable(TIME_month,WSPD_month);

WSMX_M=TT.WSMX;
WSMX_month=WSMX_M(mo,:);
TT_WSMX_mo=timetable(TIME_month,WSMX_month);

SRAD_M=TT.SRAD;
SRAD_month=SRAD_M(mo,:);
TT_SRAD_mo=timetable(TIME_month,SRAD_month);

TT_month=[TT_TAIR_mo TT_DWPT_mo TT_TAIRx_mo TT_TAIRn_mo TT_PRCP_mo TT_RELH_mo TT_PRES_mo ...
          TT_SM02_mo TT_WDIR_mo TT_WSPD_mo TT_WSMX_mo TT_SRAD_mo];

end