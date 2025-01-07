clear
format long

site='HCKM';

fileMES1='01-Nov-2009_31-Jul-2023_HCKM_daily.mat';

%load Mesonet, Climate Data and Thresholds
dircli1='/Users/erappin/Documents/Mesonet/ClimateIndices/cliSITES/';
dircli=strcat(strcat(dircli1,site,'/'));
filethresh=strcat(dircli,strcat(site,"_CLIthresh_daily"));
load(filethresh);
filemes=strcat(dircli1,fileMES1);
load(filemes);

%TT=TT_dailyMES;
TIME=TT_dailyMES.TimestampCollected;
sTIME=TIME(1);
eTIME=TIME(end);
[sY,~,~]=ymd(sTIME);
[eY,~,~]=ymd(eTIME);
Yvalues = (datetime(sY,01,01):days(1):datetime(eY,12,31))';
TT_dailyMES = retime(TT_dailyMES,Yvalues,'fillwithconstant','Constant',0);
%isD=find(all(ismember(TT_dailyMES.TimestampCollected,sTIME),2));
%ieD=find(all(ismember(TT_dailyMES.TimestampCollected,eTIME),2));

TIME_full=TT_dailyMES.TimestampCollected;
isD=find(all(ismember(TIME_full,sTIME),2));
ieD=find(all(ismember(TIME_full,eTIME),2));


YEAR=unique(TIME_full.Year);
myStructM.year=YEAR;
MNTH=["Jan" "Feb" "Mar" "Apr" "May" "Jun" "Jul" "Aug" "Sep" "Oct" "Nov" "Dec"]';
myStructM.month=MNTH;
VAR=["TAIR_month" "DWPT_month" "TAIRx_month" "TAIRn_month" "PRCP_month" "RELH_month" ...
     "PRES_month" "SM02_month" "WDIR_month" "WSPD_month" "WSMX_month" "SRAD_month"]';
myStructM.var=VAR;

ny=numel(YEAR);
nm=numel(MNTH);
nv=numel(VAR);

Md=cell(nm,nv);
My=cell(nm,nv);

yoD = year(TIME_full);         
moD = month(TIME_full);        

for i=1:nm
    im = ismember(moD,i);
    TIME_month=TIME_full(im,:);
    for j=1:nv
        TT=getData_month(TT_dailyMES,im,TIME_full);
        TTcal=TT.TIME_month;
        TTyears = year(TTcal);
        [~,TTindYears] = findgroups(TTyears);
        Vartt=TT{:,j};
        TTbyYear = arrayfun(@(y){Vartt(TTyears==y,:)},TTindYears);
        for k = 1:numel(TTbyYear)
            TTy{:,k}=TTbyYear{k};
        end
        Md{i,j}=TTy;
        My{i,j}=TTindYears;
    end
end

myStructM.year=My;
myStructM.data=Md;

DATA=myStructM.data;
MNTH=myStructM.month;
YEAR=string(myStructM.year{1,1});

IndicesY=ones(ny,1);
for i=1:ny
    yearT(i)=datetime(double(YEAR(i)),01,01);
    IndicesY(i)=datefind(yearT(i),TIME_full);
end

return

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