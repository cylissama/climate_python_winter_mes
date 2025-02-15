clear
format long

site='BMTN';
fileMES1='01-Mar-2014_01-Aug-2023_BMTN_daily';
dateS=datetime(2015,01,01);
dateE=datetime(2022,12,31);
%dateS=datetime(2010,01,01);
%dateE=datetime(2022,12,31);

%load Mesonet, Climate Data and Thresholds
dircli1='/Volumes/Mesonet/winter_break/CCdata/';
dircli=strcat(strcat(dircli1,site,'/'));
filethresh=strcat(dircli,strcat(site,"_CLIthresh_daily"));
load(filethresh);
filemes=strcat(dircli,fileMES1);
load(filemes);

%TIME_full=TT_dailyMES.TimestampCollected;
%sTIME_full=TIME_full(1);
%eTIME_full=TIME_full(end);
%YEAR_full=unique(TIME_full.Year);
%[sYf,sMf,sDf]=ymd(TIME_full(1));
%[eYf,eMf,eDf]=ymd(TIME_full(end));
%dateS=datetime(sYf+1,01,01);
%if dateS >= sTIME_full
%   isD=find(all(ismember(TIME_full,dateS),2));
%end
%dateE=datetime(eYf-1,12,31); 
%if dateE <= eTIME_full
%   ieD=find(all(ismember(TIME_full,dateE),2));
%end

TIME_full=TT_dailyMES.TimestampCollected;
sTIME_full=TIME_full(1);
eTIME_full=TIME_full(end);
[sYf,sMf,sDf]=ymd(TIME_full(1));
[eYf,eMf,eDf]=ymd(TIME_full(end));
isD=find(all(ismember(TIME_full,dateS),2));
ieD=find(all(ismember(TIME_full,dateE),2));

VARold=["TAIR" "TAIRx" "TAIRn" "DWPT" "PRCP" "PRES" "RELH" "SM02" "SRAD" "WDIR" "WSPD" "WSMX"];
VAR=["TAIR_annual" "TAIRx_annual" "TAIRn_annual" "DWPT_annual" "PRCP_annual" "PRES_annual" ... 
     "RELH_annual" "SM02_annual" "SRAD_annual" "WDIR_annual" "WSPD_annual" "WSMX_annual"]';
TT=TT_dailyMES(isD:ieD,:);
TT=removevars(TT,{'SM20' 'SM04' 'SM40' 'SM08' 'ST02' 'ST20' 'ST04' 'ST40' 'ST08' 'WSMN'});
TT=renamevars(TT,VARold,VAR);

TIME=TT.TimestampCollected;
YEAR=unique(TIME.Year);
DATAannual.year=YEAR;
DATAannual.var=VAR;

ny=numel(YEAR);
nv=numel(VAR);
DATAyear=cell(ny,1);

for k=1:ny
    TTa=TT;
    TTa(year(TIME)~=YEAR(k),:)=[];
    DATAyear{k,1}=TTa;
end
DATAannual.data=DATAyear;