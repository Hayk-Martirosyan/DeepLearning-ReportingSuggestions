#!/usr/bin/python3
import argparse

from keras.models import Sequential
from keras import optimizers
import numpy
import io
import random
import datetime
# import sqlite3
import nnModel
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras import backend as K
import keras_helpers
import math
import sys
from flask import Flask
from flask import send_file, make_response, send_from_directory,request
from flask import jsonify
import datetime

import tensorflow as tf
import sqlite3
from flask_cors import CORS

from sklearn.metrics import classification_report, confusion_matrix
numpy.set_printoptions(suppress=True,linewidth=numpy.nan,threshold=numpy.nan)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", required=True,
        help="type of processing", choices=['train', 'test', 'predict', 'evaluate', 'predict-service'])
ap.add_argument("-id", "--modelid", required=True,
            help="modelid")
FLAGS, unparsed = ap.parse_known_args()



modelId = FLAGS.modelid

ROW_MAX_PATTERN_LENGTH = 8
COL_MAX_PATTERN_LENGTH = 10

ALL_OBJECTS_ID_NAME = {
      "200":"Project",
      "201":"Project Status",
      "202":"Region",
      "203":"Country",
      "204":"Estimated Thematic Area",
      "205":"Estimated Sub-Thematic Area",
      "206":"Sub-Division 1",
      "207":"Sub-Division 2",
      "208":"Sub-Division 3",
      "209":"Context",
      "210":"Common Approach",
      "211":"Source of Fund (SOF)",
      "212":"Draft Cross-cutting Themes",
      "213":"SCA Humanitarian Flexible Funding?",
      "214":"Activity",
      "215":"Actual Thematic Area",
      "216":"Actual Sub-Thematic Area",
      "217":"Procurement Required?",
      "218":"Activity Type",
      "219":"Activity Name",
      "220":"Humanitarian Response Code",
      "221":"Humanitarian Response Category",
      "254":"Task",
      "256":"Activity Person Responsible",
      "257":"SC Person Responsible",
      "258":"Procurement Line Item",
      "262":"Unit",
      "263":"Procurement SOF",
      "264":"Procurement Procedure",
      "265":"Waiver Required",
      "266":"Partners",
      "267":"Responsible Person for Managing the Procurement Process",
      "268":"Reason For Waiver",
      "269":"HR Plan ID",
      "270":"Planned Job Title",
      "271":"Employee Type",
      "272":"HR Management",
      "273":"Donor Approval Required For",
      "274":"Related Project",
      "275":"Budget Item",
      "276":"Code - Account Name",
      "277":"Unit Type",
      "278":"Cost Center",
      "279":"Budget Year",
      "280":"Budget Month",
      "281":"Thematic Activity",
      "282":"Cost Centers",
      "283":"Cross-cutting Themes",
      "284":"Action Status",
      "285":"Employee",
      "286":"Implementing Party",
      "287":"Source of Action",
      "288":"Project Actions",
      "289":"Action Partner",
      "290":"Action Activity",
      "291":"MEAL Plan",
      "292":"Budget SOF",
      "293":"Activity Actions",
      "294":"Project Partner",
      "295":"Partner",
      "296":"Agreement Status",
      "297":"Agreement Created?",
      "298":"Agreement Currency",
      "299":"Contract Type",
      "300":"LogFrame Level 1",
      "301":"LogFrame Level 2",
      "302":"LogFrame Level 3",
      "303":"LogFrame Level 4",
      "304":"LogFrame Level 5",
      "305":"LogFrame Level 6",
      "306":"LogFrame Level 7",
      "307":"LogFrame Level 8",
      "308":"Indicator",
      "309":"ME",
      "320":"Year",
      "321":"Month",
      "322":"Semester",
      "323":"Quarter",
      "324":"Week",
      "325":"Day",
      "400":"SOF",
      "417":"Project DRC",
      "427":"SOF DRC",
      "429":"Active Projects",
      "430":"Payment",
      "431":"Project Created By",
      "432":"Activity Created By",
      "433":"Project Updated By",
      "434":"Activity Updated By",
      "435":"Budget Item Updated By",
      "500":"Individual",
      "503":"Disability",
      "504":"Nationality",
      "505":"Vulnerability",
      "506":"Marital Status",
      "507":"Registration Region",
      "508":"Registration Country",
      "509":"Registration Sub-Division 1",
      "510":"Registration Sub-Division 2",
      "511":"Registration Sub-Division 3",
      "522":"Cohort",
      "600":"Household",
      "601":"Household Head",
      "602":"Household Vulnerability",
      "603":"Household Region",
      "604":"Household Country",
      "605":"Household Sub-Division 1",
      "606":"Household Sub-Division 2",
      "607":"Household Sub-Division 3",
      "622":"Household Cohort",
      "623":"Household Individuals",
      "700":"Community",
      "701":"Community Individuals",
      "702":"Community Households",
      "707":"Community Region",
      "708":"Community Country",
      "709":"Community Sub-Division 1",
      "710":"Community Sub-Division 2",
      "711":"Community Sub-Division 3",
      "800":"Association",
      "801":"Association Individuals",
      "802":"Association Households",
      "807":"Association Region",
      "808":"Association Country",
      "809":"Association Sub-Division 1",
      "810":"Association Sub-Division 2",
      "811":"Association Sub-Division 3",
      "822":"Association Cohort",
      "2074":"Person Responsible MP",
      "58747":"Means of Verification / Data Collection Tool",
      "58748":"Income level",
      "58749":"scarface",
      "58751":"Person Responsible for Reporting",
      "58752":"Disaggregation  Management",
      "58754":"Person Responsible for Reporting",
      "58755":"Country",
      "58756":"Age",
      "58757":"Location",
      "58758":"Age Group",
      "58759":"Gender",
      "1801":"Start Date",
      "1802":"End Date",
      "1803":"Comment",
      "1804":"Modified On",
      "1805":"Created On",
      "1806":"Modified By",
      "1807":"Created By",
      "1808":"Person Responsible",
      "1809":"Employee Name",
      "1810":"Duration",
      "1811":"Progress (%)",
      "1812":"Lead Time",
      "1813":"Unit Cost",
      "1814":"No. of Units",
      "1815":"# of Actions",
      "2000":"Project Name",
      "2001":"# of Projects",
      "2002":"Status name",
      "2003":"# of Statuses",
      "2004":"Region",
      "2005":"# of Regions",
      "2006":"Country",
      "2007":"# of Countries",
      "2008":"Estimated Thematic Area",
      "2009":"# of Estimated Thematic Areas",
      "2010":"Estimated Sub-Thematic Area",
      "2011":"# of Estimated Sub-Thematic Areas",
      "2012":"Sub-Division 1",
      "2013":"# of Sub-Divisions 1",
      "2014":"Sub-Division 2",
      "2015":"# of Sub-Divisions 2",
      "2016":"Sub-Division 3",
      "2017":"# of Sub-Divisions 3",
      "2018":"Context name",
      "2019":"# of Contexts",
      "2021":"Start Date",
      "2022":"End Date",
      "2023":"Common Approach",
      "2024":"# of Common Approach",
      "2025":"Source of Fund name",
      "2026":"# Source of Fund",
      "2027":"Draft Cross-cutting Theme Name",
      "2028":"# of Draft Cross-cutting Themes",
      "2029":"Settlement Urban",
      "2030":"Description",
      "2031":"Settlement Rural",
      "2032":"SCA Humanitarian Flexible Funding?",
      "2033":"# of SCA Humanitarian Flexible Fundings",
      "2034":"Code",
      "2035":"Activity",
      "2036":"# of Activities",
      "2037":"Actual Thematic Area",
      "2038":"# of Actual Thematic Areas",
      "2039":"Actual Sub-Thematic Area",
      "2040":"# of Actual Sub-Thematic Areas",
      "2041":"Activity Comment",
      "2042":"Procurement Required?",
      "2043":"# of Procurement Required",
      "2044":"Activity Type",
      "2045":"# of Activity Types",
      "2046":"Activity Name",
      "2047":"# of Activity Names",
      "2048":"Modified On",
      "2049":"Created On",
      "2050":"Modified By",
      "2051":"Created By",
      "2052":"Activity Modified On",
      "2053":"Activity Created On",
      "2054":"Activity Modified By",
      "2055":"Activity Created By",
      "2056":"Activity Start Date",
      "2057":"Activity End Date",
      "2058":"Title",
      "2059":"# of Tasks",
      "2060":"Humanitarian Response Code",
      "2061":"# of HumanitarianResponseCode",
      "2062":"Humanitarian Response Category",
      "2063":"# of HumanitarianResponseCategory",
      "2064":"Task Comment",
      "2065":"Task Start Date",
      "2066":"Task End Date",
      "2067":"Activity Person Responsible name",
      "2068":"# of Activity Person Responsible",
      "2069":"SC Person Responsible name",
      "2070":"# of SC Person Responsible",
      "2071":"Task Progress (%)",
      "2075":"Person Responsible Name",
      "2076":"# of Person Responsible",
      "2077":"Procurement Line Item Name",
      "2078":"# of Procurement Line Items",
      "2079":"Procurement Category 1 Name",
      "2080":"# of Procurement Category 1",
      "2081":"Procurement Category 2 Name",
      "2082":"# of Procurement Category 2",
      "2083":"Procurement Category 3 Name",
      "2084":"# of Procurement Category 3",
      "2085":"Unit Name",
      "2086":"# of Units",
      "2087":"Quantity",
      "2088":"Unit Cost",
      "2089":"Estimated Cost",
      "2090":"Estimated Transport Cost",
      "2091":"Estimated Total Cost",
      "2092":"Procurement SOF Name",
      "2093":"# of Procurement SOFs",
      "2094":"Procurement Procedure Name",
      "2095":"# of Procurement Procedures",
      "2096":"Waiver Required Name",
      "2097":"# of Waiver Required",
      "2098":"Required Delivery Date of Good/Service",
      "2099":"Lead Time from Raising PR to Items Delivered",
      "2100":"Date the Purchase Request is Required",
      "2101":"Date Purchase Request Issued",
      "2102":"Partner Name",
      "2103":"# of Partners",
      "2104":"Responsible Person for Managing the Procurement Process Name",
      "2105":"# of Responsible Person for Managing the Procurement Processes",
      "2106":"Delivery Date",
      "2107":"Procurement Comment",
      "2108":"Remarks",
      "2109":"Reason For Waiver Name",
      "2110":"# of Reason For Waivers",
      "2111":"HR Plan ID Name",
      "2112":"# of HR Plan IDs",
      "2113":" Planned Job Title name",
      "2114":"# of Planned Job Titles",
      "2115":"Key Responsibilities",
      "2116":"Employee Type name",
      "2117":"# of Employee Types",
      "2118":"First Name",
      "2119":"Last Name",
      "2120":"Required Start Date",
      "2121":"Required End Date",
      "2122":"Date to Start Recruitment",
      "2123":"Remotely Managed Project?",
      "2124":"Project Duration",
      "2125":"Activity Duration",
      "2126":"Lead Time",
      "2127":"HR Management name",
      "2128":"# of HR Managements",
      "2129":" Donor Approval Required For name",
      "2130":"# of Donor Approval Required For",
      "2131":" Description of Donor Requirement",
      "2132":"Task Duration",
      "2133":"Activity Progress (%)",
      "2134":"Related Project name",
      "2135":"# of Related Projects",
      "2136":"Budget Item Name",
      "2137":"# of Budget Items",
      "2138":"Code - Account Name",
      "2139":"# of Code - Account Names",
      "2140":"Unit Type Name",
      "2141":"# of Unit Types",
      "2142":"Cost Center Name",
      "2143":"# of Cost Centers",
      "2144":"No. of Unit",
      "2145":"Quantity / Duration",
      "2146":"Unit Cost",
      "2147":"LOE (%)",
      "2148":"Commentary / Additional information",
      "2149":"Total Cost",
      "2150":"Budget Year Name",
      "2151":"# of Budget Years",
      "2152":"Budget Month Name",
      "2153":"# of Budget Months",
      "2154":"Cost",
      "2155":"Procurement not executed by Supply Chain Team",
      "2156":"Donor Requires Approval to Changes",
      "2157":"Recruitment Required?",
      "2158":"Recruitment Complete?",
      "2159":"Role requires donor approval",
      "2160":"Thematic Activity name",
      "2161":"# of Thematic Activities",
      "2162":"Budget Item Modified by",
      "2163":"Budget Item Modified on",
      "2164":"Related Project Comment",
      "2165":"Service Start Date",
      "2166":"Service End Date",
      "2167":"Cost Centers name",
      "2168":"# Cost Centers",
      "2169":"Cross-cutting Theme Name",
      "2170":"# of Cross-cutting Themes",
      "2171":"Action Status Name",
      "2172":"# of Action Statuses",
      "2173":"Employee Name",
      "2174":"# of Action Employees",
      "2175":"Implementing Party Name",
      "2176":"# of Implementing Parties",
      "2177":"Source of Action Name",
      "2178":"# of Source of Actions",
      "2179":"Project Action Name",
      "2180":"# of Project Actions",
      "2181":"Action Partner Name",
      "2182":"# of Action Partners",
      "2183":"Action Responsible Person",
      "2184":"Date of Issuance",
      "2185":"Deadline for Action",
      "2186":"Other Source of Action",
      "2187":"Completion Date",
      "2188":"Source Details",
      "2189":"Action Activity Name",
      "2190":"# of Action Activities",
      "2191":"Issue that Requires Action",
      "2192":"Action Comments",
      "2193":"MEAL Plan Name",
      "2194":"# of MEAL Plans",
      "2195":"Team Organogram",
      "2196":"Monitoring of Activities and Outputs",
      "2197":"Monitoring of Project Outcomes",
      "2198":"Monitoring Program Quality",
      "2199":"Accountability to Communities",
      "2200":"Evaluation and Research",
      "2201":"Information Systems Management",
      "2202":"Team Member Name",
      "2203":"Additional Notes",
      "2204":"Budget SOF Name",
      "2205":"# of Budget SOFs",
      "2206":"Activity Action Name",
      "2207":"# of Activity Actions",
      "2208":"Project Partner Name",
      "2209":"# of Project Partners",
      "2210":"Partner Name",
      "2211":"# of Partners",
      "2212":"Agreement Status Name",
      "2213":"# of Agreement Statuses",
      "2214":"Agreement Created Name",
      "2215":"# of Agreement Created",
      "2216":"Agreement Currency Name",
      "2217":"# of Agreement Currencies",
      "2218":"Contract Type Name",
      "2219":"# of Contract Types",
      "2220":"Budgeted Amount",
      "2221":"Payment Name",
      "2222":"# of Payments",
      "2223":"Payment Amount",
      "2224":"Planned Payment Date",
      "2225":"Planned Financial Report Date",
      "2226":"Partner Comments",
      "3100":"RFLevel1Id",
      "3101":"Actual",
      "3102":"Target",
      "3103":"Baseline",
      "3104":"IndicatorId",
      "3105":"Achievement",
      "3106":"Indicator Progress",
      "3107":"Indicator Progress(%)",
      "3200":"Year Name",
      "3201":"# of Years",
      "3202":"Month Name",
      "3203":"# of Months",
      "3204":"Semester Name",
      "3205":"# of Semesters",
      "3206":"Quarter Name",
      "3207":"# of Quarters",
      "3208":"Week Name",
      "3209":"# of Weeks",
      "3210":"Day Name",
      "3211":"# of Days",
      "4000":"SOF Name",
      "4001":"# of SOFs",
      "4034":"Project DRC name",
      "4035":"# of Project DRCs",
      "4057":"Project DRC Description",
      "4058":"Project Budget Line Flex(%)",
      "4059":"Estimated Cost",
      "4060":"Variance",
      "4061":"SOF DRC name",
      "4062":"# of SOF DRCs",
      "4063":"SOF DRC Description",
      "4064":"SOF Budget Line Flex(%)",
      "4065":"Description",
      "4066":"Budget Line Flex(%)",
      "4067":"Project Created By first name",
      "4068":"# of Project Created By",
      "4069":"Project Created By last name",
      "4070":"Activity Created By first name",
      "4071":"# of Activity Created By",
      "4072":"Activity Created By last name",
      "4073":"Project Updated By first name",
      "4074":"# of Project Updated By",
      "4075":"Project Updated By last name",
      "4076":"Activity Updated By first name",
      "4077":"# of Activity Updated By",
      "4078":"Activity Updated By last name",
      "4079":"Budget Item Updated By first name",
      "4080":"# of Budget Item Updated By",
      "4081":"Budget Item Updated By last name",
      "5000":"Individual Name",
      "5001":"# of Individuals",
      "5006":"Date of Birth",
      "5007":"Disability name",
      "5008":"# of Disabilities",
      "5009":"Nationality name",
      "5010":"# of Nationalities",
      "5011":"Social Security National ID",
      "5012":"Mobile Number",
      "5013":"Vulnerability name",
      "5014":"# of Vulnerabilities",
      "5015":"Ethnicity",
      "5016":"Marital Status name",
      "5017":"# of Marital Statuses",
      "5018":"Notes",
      "5019":"Registration Region name",
      "5020":"# of Registration Regions",
      "5021":"Registration Country name",
      "5022":"# of Registration Countries",
      "5023":"Registration Sub-Divisions 1 name",
      "5024":"# of Registration Sub-Divisions 1",
      "5025":"Registration Sub-Divisions 2 name",
      "5026":"# of Registration Sub-Divisions 2",
      "5027":"Registration Sub-Divisions 3 name",
      "5028":"# of Registration Sub-Divisions 3",
      "5029":"Registration Address",
      "5051":"Alias",
      "5052":"Cohort name",
      "5053":"# of Cohorts",
      "5054":"Passport No",
      "5055":"UNHCR No",
      "5152":"Age",
      "6000":"Household Name",
      "6001":"# of Households",
      "6002":"Household Head name",
      "6003":"# of Household Heads",
      "6004":"Vulnerability name",
      "6005":"# of Household Vulnerabilities",
      "6006":"Household Region name",
      "6007":"# of Household Regions",
      "6008":"Household Country name",
      "6009":"# of Household Countries",
      "6010":"Household Sub-Divisions 1 name",
      "6011":"# of Household Sub-Divisions 1",
      "6012":"Household Sub-Divisions 2 name",
      "6013":"# of Household Sub-Divisions 2",
      "6014":"Household Sub-Divisions 3 name",
      "6015":"# of Household Sub-Divisions 3",
      "6016":"Special Support Needs",
      "6017":"Household Estimated Size",
      "6018":"Household Code",
      "6019":"Household Address",
      "6052":"Household Cohort name",
      "6053":"# of Household Cohorts",
      "6054":"Household Individual name",
      "6055":"# of Household Individuals",
      "7000":"Community Name",
      "7001":"# of Communities",
      "7002":"Community Individual name",
      "7003":"# of Community Individuals",
      "7004":"Community Household name",
      "7005":"# of Community Households",
      "7006":"Community Code",
      "7019":"Community Region name",
      "7020":"# of Community Regions",
      "7021":"Community Country name",
      "7022":"# of Community Countries",
      "7023":"Community Sub-Divisions 1 name",
      "7024":"# of Community Sub-Divisions 1",
      "7025":"Community Sub-Divisions 2 name",
      "7026":"# of Community Sub-Divisions 2",
      "7027":"Community Sub-Divisions 3 name",
      "7028":"# of Community Sub-Divisions 3",
      "7029":"Community Address",
      "7030":"Community Estimated Size",
      "8000":"Association Name",
      "8001":"# of Associations",
      "8002":"Association Individual name",
      "8003":"# of Association Individuals",
      "8004":"Association Household name",
      "8005":"# of Association Households",
      "8006":"Association Code",
      "8019":"Association Region name",
      "8020":"# of Association Regions",
      "8021":"Association Country name",
      "8022":"# of Association Countries",
      "8023":"Association Sub-Divisions 1 name",
      "8024":"# of Association Sub-Divisions 1",
      "8025":"Association Sub-Divisions 2 name",
      "8026":"# of Association Sub-Divisions 2",
      "8027":"Association Sub-Divisions 3 name",
      "8028":"# of Association Sub-Divisions 3",
      "8029":"Association Address",
      "8030":"Association Estimated Size",
      "8052":"Association Cohort name",
      "8053":"# of Association Cohorts",
      "58791":"Comment",
      "58792":"Target ENG",
      "58805":"Baseline ENG",
      "58815":"Actual ENG",
      "58823":"Baseline ENG",
      "58824":"Target",
      "58845":"Baseline",
      "58846":"Target ",
      "58847":"Actual ",
      "58853":"Verification Plan",
      "58854":"Actual Once Eng",
      "58855":"Actual Quarterly Eng",
      "58867":"Actual ENG",
      "58872":"Baseline",
      "58874":"Source of Data",
      "58889":"Persona Responsible for Reporting",
      "58890":"Data Limitation",
      "58891":"Baseline Collection Date",
      "58893":"Baseline ENG",
      "58896":"Target ENG",
      "58914":"None Value List",
      "58915":"None Comment",
      "58917":"Date",
      "58918":"Baseline",
      "58920":"Target",
      "58922":"Expression",
      "58927":"Baseline",
      "58928":"Actual_ENG",
      "58932":"Comment",
      "58934":"Baseline ENG",
      "58947":"Target_ENG",
      "58953":"Means of verification/Data Collection Tool",
      "58955":"Baseline ENG",
      "58956":"Target ENG",
      "58957":"Actual ENG",
      "58958":"Baseline",
      "58959":"Verification Pla",
      "58962":"Data Limitation",
      "58963":"Target Eng",
      "58964":"Percent Eng",
      "58966":"Actual Annually Eng",
      "58972":"Means of Verification ENG",
      "58973":"Data Source ENG",
      "58975":"MP",
      "58977":"Comment Eng",
      "58998":"Actual",
      "58999":"Target ENG",
      "59000":"Actual ENG",
      "59003":"Actual",
      "59004":"Verification Plan",
      "59005":"Data Limitation",
      "59006":"Means of Verification / Data Collection Tool",
      "59009":"Means of verification/Data Collection Tool",
      "59014":"Validation save",
      "59015":"Indicator for new born children  DP",
      "59018":"Data Limitations",
      "59030":"Baseline Comment",
      "59031":"Target Comment",
      "59032":"Actual Comment",
      "59033":"Actual",
      "59035":"Actual",
      "59036":"Comment",
      "59052":"Baseline",
      "59053":"Target",
      "59054":"Actual",
      "59067":"baseline %",
      "59068":"target %",
      "59069":"Actual %",
      "59071":"Actual ENG",
      "59077":"Baseline",
      "59080":"Meal",
      "59081":"Target",
      "59082":"Actual",
      "59097":"Unit of Measure",
      "59123":"Target",
      "59124":"Actual",
      "59127":"Baseline",
      "59160":"Target",
      "59179":"Baseline",
      "59193":"Actual",
      "59194":"Comment",
      "59228":"Planned Amount",
      "59230":"Actual Amount Date",
      "59231":"Planned Amount Date",
      "59232":"Actual Amount",
      "59294":"Baseline",
      "59295":"Actual",
      "59305":"% of Targets",
      "59316":"Percent",
      "59362":"Baseline",
      "59372":"Target",
      "429":"Active Projects",
      "1801":"Start Date",
      "1802":"End Date",
      "1803":"Comment",
      "1804":"Modified On",
      "1805":"Created On",
      "1806":"Modified By",
      "1807":"Created By",
      "1808":"Person Responsible",
      "1809":"Employee Name",
      "1810":"Duration",
      "1811":"Progress (%)",
      "1812":"Lead Time",
      "1813":"Unit Cost",
      "1814":"No. of Units",
      "1815":"# of Actions",
      "4065":"Description",
      "4066":"Budget Line Flex(%)",
      "5152":"Age",
      "4060":"Variance",
      "502":"Gender",
      "5004":"Gender name",
      "5005":"# of Genders",
      "58747":"Means of Verification / Data Collection Tool",
      "58748":"Income level",
      "58749":"scarface",
      "58751":"Person Responsible for Reporting",
      "58752":"Disaggregation  Management",
      "58753":"Armen Test Disaggregation",
      "58754":"Person Responsible for Reporting",
      "58755":"Country",
      "58756":"Age",
      "58757":"Location",
      "58758":"Age Group",
      "58759":"Gender",
      "58790":"aaa",
      "58791":"Comment",
      "58792":"Target ENG",
      "58805":"Baseline ENG",
      "58815":"Actual ENG",
      "58823":"Baseline ENG",
      "58824":"Target",
      "58834":"Data Source",
      "58845":"Baseline",
      "58846":"Target ",
      "58847":"Actual ",
      "58853":"Verification Plan",
      "58854":"Actual Once Eng",
      "58855":"Actual Quarterly Eng",
      "58856":"Target_test",
      "58867":"Actual ENG",
      "58872":"Baseline",
      "58874":"Source of Data",
      "58888":"Actual Monthly Eng",
      "58890":"Data Limitation",
      "58891":"Baseline Collection Date",
      "58893":"Baseline ENG",
      "58896":"Target ENG",
      "58914":"None Value List",
      "58915":"None Comment",
      "58917":"Date",
      "58918":"Baseline ",
      "58919":"Commentt B",
      "58920":"Target ",
      "58921":"Actual required ",
      "58922":"Expression",
      "58927":"Baseline",
      "58928":"Actual_ENG",
      "58932":"Comment",
      "58934":"Baseline ENG",
      "58947":"Target_ENG",
      "58953":"Means of verification/Data Collection Tool",
      "58955":"Baseline ENG",
      "58956":"Target ENG",
      "58957":"Actual ENG",
      "58958":"Baseline",
      "58959":"Verification Pla",
      "58960":"Test for ref sheet",
      "58962":"Data Limitation",
      "58963":"Target Eng",
      "58964":"Percent Eng",
      "58966":"Actual Annually Eng",
      "58968":"Date ",
      "58970":"Rationale ENG",
      "58972":"Means of Verification ENG",
      "58973":"Data Source ENG",
      "58975":"MP",
      "58977":"Comment Eng",
      "58998":"Actual",
      "58999":"Target ENG",
      "59000":"Actual ENG",
      "59003":"Actual",
      "59004":"Verification Plan",
      "59005":"Data Limitation",
      "59006":"Means of Verification / Data Collection Tool",
      "59009":"Means of verification/Data Collection Tool",
      "59015":"Indicator for new born children  DP",
      "59018":"Data Limitations",
      "59025":"MEAL DATE",
      "59026":"MEAL TEXT",
      "59027":"MEAL VALUE LIST",
      "59030":"Baseline Comment",
      "59031":"Target Comment",
      "59032":"Actual Comment",
      "59033":"Actual",
      "59035":"Actual",
      "59036":"Comment",
      "59040":"Means of verification/Data Collection Tool",
      "59052":"Baseline",
      "59053":"Target",
      "59054":"Actual",
      "59058":"Actual",
      "59067":"baseline %",
      "59068":"target %",
      "59069":"Actual %",
      "59071":"Actual ENG",
      "59077":"Baseline",
      "59080":"Meal",
      "59081":"Target",
      "59082":"Actual",
      "59094":"Target",
      "59097":"Unit of Measure",
      "59112":"Basline",
      "59123":"Target",
      "59124":"Actual",
      "59127":"Baseline",
      "59160":"Target",
      "59179":"Baseline",
      "59193":"Actual",
      "59194":"Comment",
      "59209":"Narrative",
      "59210":"Value",
      "59211":"Comment",
      "59228":"Planned Amount",
      "59230":"Actual Amount Date",
      "59231":"Planned Amount Date",
      "59232":"Actual Amount",
      "59248":"Means of verification/Data Collection Tool",
      "59251":"Justification",
      "59252":"Source of data",
      "59254":"MEAL PIRS Attributes_text",
      "59256":"Person Responsible for Data Collection",
      "59258":"MEAL PIRS Attributes_number",
      "59261":"Means of verification/Data Collection Tooltext",
      "59262":"Comment",
      "59282":"% of Actual",
      "59291":"Percent",
      "59294":"Baseline",
      "59295":"Actual",
      "59305":"% of Targets",
      "59316":"Percent",
      "59362":"Baseline",
      "59372":"Target",
      "59393":"Person Responsible for Reporting",
      "59394":"Data limitations",
      "59407":"MEAL PIRS Attributes",
      "59408":"MEALS PIRS Attributes",
      "59409":"MEAL PIRS Attributes",
      "58747":"Means of Verification / Data Collection Tool",
      "58749":"scarface",
      "58751":"Person Responsible for Reporting",
      "58752":"Disaggregation  Management",
      "58753":"Armen Test Disaggregation",
      "58754":"Person Responsible for Reporting",
      "58755":"Country",
      "58756":"Age",
      "58757":"Location",
      "58758":"Age Group",
      "58759":"Gender"
}

ALL_OBJECTS={}
for id in ALL_OBJECTS_ID_NAME:
    name = ALL_OBJECTS_ID_NAME[id];
    ALL_OBJECTS[name]=id
ALL_OBJECTS_LIST = list(ALL_OBJECTS_ID_NAME.keys())
ALL_OBJECTS_LIST.sort()
print(len(ALL_OBJECTS_LIST))

# MAX_WORD_LENGTH = 18
# MIN_WORD_LENGTH = 4
# MIN_PREDICTION_LENGTH=3
# LETTER_COUNT = 26
X_SHAPE = (ROW_MAX_PATTERN_LENGTH+COL_MAX_PATTERN_LENGTH, len(ALL_OBJECTS_LIST))
Y_SHAPE = (len(ALL_OBJECTS_LIST))
# def loadWords():
#     words = []
#     with open("google-10000-english-usa.txt") as f:
#         for (i,line) in enumerate(f):
#             if len(line)>=4 and len(line)<=MAX_WORD_LENGTH:
#                 words.append(line)
#     return words

# def letterToFeature(letter):
#     feature = numpy.zeros(LETTER_COUNT);
#     feature[ord(letter)] = 1
#     return feature;

# def letterToFeatureIndex(letter):
#     return ord(letter) - 97#ord('a')

# def wordToData(letters):
#     dataX = numpy.zeros(X_SHAPE)
#     for i, letter in enumerate(letters):
#         dataX[i, letterToFeatureIndex(letter)] = 1
#     return dataX;

# def generateTrainingItem(letters, prediction):
#     dataX = wordToData(letters)
#     dataY = numpy.zeros(Y_SHAPE)
    
#     dataY[letterToFeatureIndex(prediction)] = 1
#     return dataX, dataY


# def genData2(words):
#     count = len(words)
#     x = [];
#     y = [];
#     for word in words:
#         for i in range(MIN_PREDICTION_LENGTH, len(word)-1, 1):
#             # print(word, word[:i], word[i])
#             dataX, dataY = generateTrainingItem(word[:i], word[i])
#             # print(dataX)
#             # print(dataY)
#             # print (word[:i])
#             # print(word[i+1])
#             x.append(dataX)
#             y.append(dataY);

#     x = numpy.asarray(x)
#     y = numpy.asarray(y)
#     return x, y

# def genData(size, f):
#     m = size;
#     x = numpy.zeros((m, 1));
#     y = numpy.zeros((m, 1));

#     for i in range(m):
#         xi = random.uniform(-10.0,10.0);
#         yi = f(xi)  
#         x[i]=xi;
#         y[i] = yi   
#     return x, y


def patternToArray(pattern):
    if pattern=="":
        return []
    else:
        return make_one_hot(pattern.split(","))

def mergePatterns(rowPattern, columnPattern):
    pattern = []
    pattern+=rowPattern
    pattern += [numpy.zeros(len(ALL_OBJECTS_LIST), dtype='int') 
                for _ in range(ROW_MAX_PATTERN_LENGTH-len(rowPattern))]
    pattern+=columnPattern;
    pattern += [numpy.zeros(len(ALL_OBJECTS_LIST), dtype='int') 
                for _ in range(COL_MAX_PATTERN_LENGTH-len(columnPattern))]
    return numpy.asarray(pattern);
    
def loadData():
    patterns = loadPatterns();
    X_data = []
    Y_data = []

    for pattern in patterns:#-2:-1
        # print(pattern)
        rowPatternOneHot = patternToArray(pattern[1]);
        # print(pattern[1])
        # print(recover(rowPatternOneHot))
        columnPatternOneHot = patternToArray(pattern[2]);
        if len(columnPatternOneHot)<1:
            continue
              
        # print(pattern[2])
        # print(recover(columnPatternOneHot))
        # x = mergePatterns(rowPatternOneHot, columnPatternOneHot[:-1]);
        # y = columnPatternOneHot[-1:]
        # # print(recover(x))
        # # print(recover(y))
        # X_data.append(x)
        # Y_data.append(y)

        couples = make_couples([rowPatternOneHot], [columnPatternOneHot], ROW_MAX_PATTERN_LENGTH, COL_MAX_PATTERN_LENGTH)
        # print("len c={}".format(len(couples)))
        # # print(recover(couples[0][0]))
        # print(len(couples[0][0]))
        # # x = numpy.zeros(X_SHAPE)
        # # y = numpy.zeros(Y_SHAPE)

        
        X_couples = [[element[0][:] , element[1][:-1]] for element in couples]
        Y_couples = [[element[0][-1:] , element[1][-1:]] for element in couples]
        for x_couple, y_couple in zip(X_couples, Y_couples):
            x_couple[0] += [numpy.zeros(len(ALL_OBJECTS_LIST), dtype='int') 
                for _ in range(ROW_MAX_PATTERN_LENGTH-len(x_couple[0]))]
            x_couple[1] += [numpy.zeros(len(ALL_OBJECTS_LIST), dtype='int') 
                for _ in range(COL_MAX_PATTERN_LENGTH-len(x_couple[1]))]
            
            
            # y_couple += [numpy.zeros(len(ALL_OBJECTS_LIST), dtype='int') 
            #     for _ in range(ROW_MAX_PATTERN_LENGTH-len(y_couple))]
            # if(len(x_couple[0] + x_couple[1])==16):
            #     print(recover(x_couple[0] + x_couple[1]))
            X_data.append(x_couple[0] + x_couple[1])
            Y_data.append(y_couple[1][0])#y_couple[0] + 
            # print(recover(x_couple[0] + x_couple[1]))
            # print(len(y_couple[1][0]))

        # if(len(X_couples)==0):
        #     continue



    X_data = numpy.asarray(X_data)
    Y_data = numpy.asarray(Y_data);
    # print(len(Y_data))

    Y_data = Y_data.reshape((-1, len(ALL_OBJECTS_LIST)))

    print(X_data.shape)
    print(Y_data.shape)
    # for x,y in zip(X_data[0:300], Y_data[0:300]):
    #     print("{} / {} ".format(recover(x), recover([y])))
        # print("{} ".format(recover(y)))
    # x_train, y_train = genData2(words)
    # x_test, y_test = x_train[0:100,:,:], y_train[0:100,:]
    x_cv, y_cv = numpy.expand_dims(numpy.zeros(X_SHAPE), 0), numpy.expand_dims(numpy.zeros(Y_SHAPE), 0) 
    return X_data, Y_data, X_data[0:1], Y_data[0:1], x_cv, y_cv
    # return x_train, y_train, x_test, y_test, x_cv, y_cv


def loadTestData():
    m = 1000;
    f = lambda x: math.sin(x)#x**2#10*x+5#1000*x*x#math.sin(x)
    x_train, y_train = genData(m, f)
    x_test, y_test = genData(int(m/5), f)
    x_cv, y_cv = genData(int(m/5), f)

    return x_train, y_train, x_test, y_test, x_cv, y_cv

def predict (patternArray):
    # print(patternArray)

    x_data = numpy.expand_dims(numpy.zeros(X_SHAPE), 0);
    x_data[0] = patternArray
    # print(x_data)
    y_pred = model.predict(x_data, batch_size=32)
    # print (y_pred)
    suggestions = []
    
    for i,p in enumerate(y_pred[0]):
        suggestions.append({'label':ALL_OBJECTS_ID_NAME[ALL_OBJECTS_LIST[i]], 'prediction':p, 'id':ALL_OBJECTS_LIST[i]})

    suggestions.sort(key=lambda item:item['prediction'], reverse=True)
    return suggestions


def insertPatterToDB(reportid, rowCategories, columnCategories):
    now = datetime.datetime.now()
    with sqlite3.connect('/ml/patterns.sqlite') as conn:
        c = conn.cursor()
        c.execute('insert into patterns (reportid, rowPattern, columnPattern, datetime) values(?,?,?,?)', [reportid, rowCategories, columnCategories, now.strftime("%Y-%m-%d %H:%M")])
        conn.commit()

def loadPatterns():
    with sqlite3.connect('/ml/patterns.sqlite') as conn:
        c = conn.cursor()
        patterns = c.execute('''SELECT * 
                                FROM patterns
                                order by datetime desc''').fetchall()

        top = patterns[:5]

        return top*5+patterns;

def md(rowObjects, columnObjects):
    print(rowObjects, columnObjects)
    rowCategoriesIds = []
    for rowObject in rowObjects:
        rowCategoriesIds.append(nameToId(rowObject))

    columnCategoriesIds = []
    for columnObject in columnObjects:
        columnCategoriesIds.append(nameToId(columnObject))

    insertPatterToDB(-1, ','.join(rowCategoriesIds), ','.join(columnCategoriesIds))


def nameToId(objectName):
    if objectName in ALL_OBJECTS:
        return ALL_OBJECTS[objectName]
    else:
        raise Exception("Invalid entry {}".format(objectName));


##
def make_one_hot(pattern):
    # patterns_one_hot = []
    # for i, pattern in enumerate(patterns):
        # print(pattern)
    pattern_one_hot = []
    for j, obj in enumerate(pattern):
        vec = [0 for _ in range(len(ALL_OBJECTS_LIST))]
        vec[ALL_OBJECTS_LIST.index(obj)] = 1
        pattern_one_hot.append(vec)
    # patterns_one_hot.append(pattern_one_hot)
    
    # print(patterns)
    # print(len(patterns_one_hot[0]))
    # return patterns_one_hot
    return pattern_one_hot;


def generate_splits(pattern, max_pattern_length, weight=5):
    result = [pattern[:i] for i in range(1, len(pattern)+1)] * weight
    for i in range(1, len(pattern)-1):
        result += [pattern[i:j] for j in range(i+2, len(pattern)+1)]
    # for vec in result:
    #     vec += [numpy.zeros(len(vec[0]), dtype='int') 
    #             for _ in range(max_pattern_length-len(vec))]
    return result


def populateData(patterns, max_pattern_length):
    result_data = []
    for pattern in patterns:
        result_data += generate_splits(pattern, max_pattern_length)
    # return numpy.array(result_data)
    return result_data


def make_couples(row_patterns, col_patterns, row_max_pattern_length, col_max_pattern_length):
    row_populated_data = populateData(row_patterns, row_max_pattern_length)
    # print(len(col_patterns[0]))
    col_populated_data = populateData(col_patterns, col_max_pattern_length)
    # print(len(col_populated_data))
    
    couple_data = []
    for row_pattern in row_populated_data:
        for col_pattern in col_populated_data:
            couple_data.append([row_pattern, col_pattern])
    return couple_data
    # return numpy.array(couple_data)

def recover(pattern_one_hot):
    pattern = []
    for obj_vec in pattern_one_hot:
        if max(obj_vec)==0:
            pattern.append('-')
        else:
            pattern.append(ALL_OBJECTS_LIST[numpy.argmax(obj_vec)])
    return pattern

def train(modelId, epochs, continueTraining):
    x_train, y_train, x_test, y_test, x_cv, y_cv = loadData()#loadAllData(),loadTestData

    batch_size = 32
    if  continueTraining:
        file = open("model-{}-last.epoch".format(modelId), 'r')
        initEpochs = int(file.read())
        epochs+=initEpochs
        file.close()
    else:
        initEpochs = 0
    optimizer = keras_helpers.MyAdamOptimizer(initEpochs, lr=0.001);
    # optimizer = optimizers.MyRMSpropOptimizer(initEpochs, lr=3e-6);
    # optimizer = optimizers.SGD(lr=0.0003, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = optimizers.SGD(lr=0.0003, decay=1e-6)
    # optimizer = keras_helpers.MySGDOptimizer(initEpochs, lr=1e-0001,decay=1e-06, momentum=0.9, nesterov=True)#
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[keras_helpers.top_4_accuracy])#'categorical_accuracy'
    if  continueTraining:
        model.load_weights("model-{}-last.hdf5".format(modelId), by_name=False);
    
    # callback_test = MyCallback(x_cv, y_cv);
    # callback_train = MyCallback(x_train, y_train);
    tensorboard = keras_helpers.MultiClassTensorBoard(trainSet={'x':x_train, 'y':y_train}, testSet={'x':x_test, 'y':y_test},
                                        log_dir='/ml/tlogs',write_images=True, write_grads=True, histogram_freq=50)
    dataGraphTensorBoard = keras_helpers.DataGraphTensorBoard(
                                        log_dir='/ml/tlogs/training', trainSet={'x':x_train, 'y':y_train}, byIndex = False, frequency=500)
    
    dataValidationGraphTensorBoard = keras_helpers.DataGraphTensorBoard(
                                        log_dir='/ml/tlogs/validation', trainSet={'x':x_cv, 'y':y_cv}, byIndex = False, frequency=500)
    

    history = model.fit(x_train, y_train,  batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_cv, y_cv)#, 
                        ,callbacks=[tensorboard], initial_epoch = initEpochs)# callbacks=[callback_train, callback_test])#validation_data=(x_test, y_test),dataGraphTensorBoard,tensorboard
     
    # print (model.evaluate(x_cv, y_cv, verbose=1))
    file = open("model-{}-last.epoch".format(modelId), 'w')
    file.write("{}".format((epochs)));
    file.close()
    model.save_weights("model-{}-last.hdf5".format(modelId))


if FLAGS.type=='train':
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-c", "--continue",
            help="continue from previous training")
    FLAGS, unparsed = ap.parse_known_args(unparsed)
    
    # x_train = x_train[0:100,:,:,:]
    # y_train = y_train[0:100,:]
    # print (x_train.shape)
    # print (y_train.shape)
    # featureCount = x_train.shape[1]
    print (modelId)
    model = nnModel.createModel(modelId, X_SHAPE, len(ALL_OBJECTS_LIST));
    print (model.summary())
    epochs = 50
    continueTraining = vars(FLAGS)['continue']
    train(modelId, epochs, continueTraining)
    
    # model.save_weights("model-{}-{}.hdf5".format(modelId,datetime.datetime.now()))

# elif FLAGS.type=='evaluate':
    
#     model = nnModel.createModel(modelId, X_SHAPE);
#     x_test_pos, y_test_pos = loadData(2, 1, 1500);
#     x_test_neg, y_test_neg = loadData(2, 0, 2000);
#     x_test_neg2, y_test_neg2 = loadSingleLineAsNegativeData(2, 1000);
#     # x_test_q, y_test_q =     loadData(2,  100, 1);

#     x_test = numpy.concatenate((x_test_pos, x_test_neg, x_test_neg2))
#     y_test = numpy.concatenate((y_test_pos, y_test_neg, y_test_neg2))

#     batch_size = 32
#     model.load_weights("model-{}-last.hdf5".format(modelId), by_name=False);
#     optimizer = optimizers.Adam(lr=0.000003);

#     model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
#     print(model.evaluate(x_test, y_test))
#     predictions = model.predict(x_test, batch_size=32)
#     print (calculateScore(predictions, y_test))

# elif FLAGS.type=='predict':

#     # x_train, y_train, x_test, y_test, x_cv, y_cv = loadAllData()
#     model = nnModel.createModel(modelId, X_SHAPE);
#     model.load_weights("model-{}-last.hdf5".format(modelId), by_name=False);
#     while(True):
#         line = sys.stdin.readline()
#         word = line[:len(line)-1]
#         if len(word)<MIN_PREDICTION_LENGTH or len(word)>MAX_WORD_LENGTH:
#             print("word length should be between {}-{}, actual value is {}".format(MIN_PREDICTION_LENGTH, MAX_WORD_LENGTH, len(word)))
#             continue
       
#         suggestions = predict(word)
        
#         for suggestion in suggestions[:4]:
#             print ("{} : {:.2f}".format(suggestion['label'], suggestion['prediction'])) 
    
elif FLAGS.type=='predict-service':

    # x_train, y_train, x_test, y_test, x_cv, y_cv = loadAllData()
    model = nnModel.createModel(modelId, X_SHAPE, len(ALL_OBJECTS_LIST));
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=[keras_helpers.top_4_accuracy])

    model.load_weights("model-{}-last.hdf5".format(modelId), by_name=False);

    app = Flask(__name__)
    CORS(app)
    graph = tf.get_default_graph()

    @app.route("/")
    def root():
        return "Welcome!\nusage: /prediction/min_3_letter"

    @app.route("/columnprediction/<string:row>/<string:column>", methods=['GET'])
    def columnPrediction(row, column):
        if row==",":
            row = ""
        if column==",":
            column=""
        rowPatternOneHot = patternToArray(row);
        columnPatternOneHot = patternToArray(column);

        x = mergePatterns(rowPatternOneHot, columnPatternOneHot);
        # print(len(x[15]))
        # print(recover(x))
        x = numpy.asarray(x).reshape(X_SHAPE)
        global graph
        with graph.as_default():
            suggestions = predict(x)
            
            ids = []
            debug = ""
            for suggestion in suggestions[:4]:
                if suggestion['prediction']>0.10:
                    debug += "{} : {:.2f}, ".format(suggestion['label'], suggestion['prediction'])
                    ids.append(suggestion['id'])
            response = '{"predictions":['+",".join(ids)+'],\n"debug":"'+debug+'"}';
            return response;
    
    @app.route("/rowprediction/<string:row>/<string:column>", methods=['GET'])
    def rowPrediction(row, column):
        return '{"predictions":[]}';

    @app.route("/pattern/<string:reportid>/<string:rowCategories>/<string:columnCategories>", methods=['GET','PUT'])
    def addPattern(reportid, rowCategories, columnCategories):
        if rowCategories==",":
            rowCategories = ""
        if columnCategories==",":
            columnCategories=""

        insertPatterToDB(reportid, rowCategories, columnCategories)
        return "OK"

    @app.route("/train/<int:epochs>", methods=['GET'])
    def trainService(epochs):
        global graph
        with graph.as_default():
            train(9, epochs, 1)
            return "Done"


    @app.route("/patterns", methods=['GET'])
    def getPatterns():
        with sqlite3.connect('/ml/patterns.sqlite') as conn:
            c = conn.cursor()
            patterns = c.execute('SELECT * FROM patterns').fetchall()
        return jsonify(patterns)

    app.run(debug=True, host= '0.0.0.0')

elif FLAGS.type=='test':
    # words = loadWords()
    maxlen = 0;
    loadData()
    # md(["LogFrame Level 1", "Indicator"],["Baseline", "Actual", "Target", "Achievement"])
    # md(["Indicator"],["Baseline", "Actual", "Target", "Achievement"])
    # md(["Indicator"],["Actual", "Target", "Achievement"])
    # md(["Project", "Activity"],["Start Date", "End Date", "Activity Progress (%)"])
    # md(["Activity"],["Cost"])
    # md(["Project", "Activity"],["Start Date", "End Date", "Activity Progress (%)"])
    # md(["Estimated Thematic Area", "Project"],["Total Cost", "Project Action Name", "Action Status Name", "Implementing Party Name", "Source of Action Name", "Action Comments"])
    # md(["Project"],["Actual Amount", "Planned Amount", "# of Activities", "# of Tasks"])
    # md(["Project", "Indicator"],["Actual", "Target"])
    # md(["Activity", "Task"],["Start Date", "End Date", "Task Progress (%)", "Duration", "Person Responsible"])
    # # md(["Award"],["Total Award Amount"])
    # md(["Project"],["Status name"])
    # md(["Activity","Task"], ["Person Responsible", "Person Responsible", "Person Responsible", "Person Responsible"])
    # md(["Activity", "Task"],["Start Date", "End Date", "Task Progress (%)"])
    # md(["Project Actions"], ["Completion Date"])



    # md(["Project", "Activity", "Task"],["# of Activities", "# of Tasks", "Planned Amount", "Actual Amount"])
    # md(["Activity", "Indicator"],["Actual", "Target", "Achievement"])
    # md(["Project", "Activity"],["# of Tasks", "Planned Amount", "Actual Amount", "Start Date", "Description"])
    # md(["Project", "Activity"],["Actual Amount", "Planned Amount", "# of Activities", "# of Tasks"])
    # md(["Activity", "Task"],["Start Date", "End Date", "Task Progress (%)", "Duration", "Person Responsible"])
    # md(["Indicator"],["Baseline", "Actual", "Target", "Achievement"])
    # md(["Project", "Activity", "Budget Item"],["Created By", "Created On", "Modified By", "Modified On"])
    # md(["Activity","Task"], ["Person Responsible", "Person Responsible", "Person Responsible", "Person Responsible"])
    # md(["Project", "Activity", "Task"],["# of Activities", "# of Tasks", "Planned Amount", "Actual Amount"])


    # md(["Project", "LogFrame Level 1", "Indicator", "LogFrame Level 2", "Indicator", "LogFrame Level 3", "Indicator" ],["Baseline", "Actual", "Target", "Achievement"])

    # md(["LogFrame Level 1", "Indicator", "LogFrame Level 2", "Indicator"],[])
    # md(["Project", "Activity"], ["Payment Amount", "Budgeted Amount"])


    # for word in words:
    #     if len(word)>maxlen:
    #         maxlen=len(word)
    # print(maxlen)
    # x_train, y_train, x_test, y_test, x_cv, y_cv = loadAllData()
    # data = {};
    # for x, y in zip(x_train, y_train):
    #     key = "_".join(map(str,x));
    #     value = y
    #     if key in data:
    #         l = data[key]
    #     else:
    #         l = [];
    #         data[key] = l
    #     l.append(y)
    # print (len(x_train))
    # for k, v in data.items():
    #     if len(v)>1:
    #         print (v)
    # model = nnModel.createModel(modelId);
    # model.load_weights("model-{}-last.hdf5".format(modelId), by_name=False);
    # predictions = model.predict(x_train, batch_size=32)
    # m = len(x_train)
    # z = numpy.zeros((m, 2));
    
    # z[:,0] = y_train[:, 0]
    # z[:,1] = predictions[:,0]
    # result = numpy.append(x_train, z, 1);

    # numpy.savetxt('result.csv',result, delimiter=',',fmt='%10.5f')

    # for layer in model.layers:
    #     weights = layer.get_weights()
    #     print (weights[0:10])











# md(["Indicator", "Disability"],["Total Cost", "Passport No"])


