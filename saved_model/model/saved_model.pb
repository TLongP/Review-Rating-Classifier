??
?*?*
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??
?
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	?N*
dtype0
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:@*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:@*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@	*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@	*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:	*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:	*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
l

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name2703*
value_dtype0	
~
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_103*
value_dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*,
shared_nameAdam/embedding/embeddings/m
?
/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m*
_output_shapes
:	?N*
dtype0
?
Adam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv1d/kernel/m
?
(Adam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/m*"
_output_shapes
:@*
dtype0
|
Adam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv1d/bias/m
u
&Adam/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@	*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:@	*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:	*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:	*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*,
shared_nameAdam/embedding/embeddings/v
?
/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v*
_output_shapes
:	?N*
dtype0
?
Adam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv1d/kernel/v
?
(Adam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/v*"
_output_shapes
:@*
dtype0
|
Adam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv1d/bias/v
u
&Adam/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@	*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:@	*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:	*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:	*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
ڤ
Const_4Const*
_output_shapes	
:?N*
dtype0*??
value??B???NBtheBandBaBofBtoBisBinBitBiBthisBthatBbrBwasBasBwithBforBmovieBbutBfilmBonBnotByouBareBhisBhaveBbeBheBoneBitsBatBallBbyBanBtheyBfromBwhoBsoBlikeBjustBorBherBaboutBifBhasBoutBsomeBthereBwhatBgoodBveryBwhenBmoreBmyBevenBnoBupBwouldBsheBtimeBonlyBreallyBtheirBwhichBseeBwereBhadBstoryBcanBmeBweBthanBmuchBwellBbeenBgetBwillBdoBgreatBbadBalsoBotherBintoBbecauseBpeopleBhowBmostBfirstBhimBdontBthenBmoviesBmadeBmakeBthemBcouldBfilmsBwayBanyBtooB
charactersBafterBthinkBwatchBmanyBseenBbeingBtwoB	characterBneverBactingBloveBwhereBplotBdidBknowBlittleBbestBshowBeverBlifeBdoesByourBbetterBoffBoverBstillBsayBendBtheseBsceneBwhyBscenesBwhileBmanBhereBsuchB	somethingBgoBshouldBthroughBimBthoseBbackBrealBwatchingBdoesntBthingBnowBdidntBactorsByearsBthoughBactuallyBanotherBmakesBfunnyBfindBnothingBbeforeBlookBgoingBsameBlotBnewBworkBeveryBfewBoldBusBpartBcantBdirectorBthatsBagainBwantBthingsBquiteBcastBprettyBseemsBgotBdownByoungBtakeBaroundBhoweverBworldBfactBenoughBbigBthoughtBgiveBbothBhorrorBiveBmayBbetweenBownBlongBwithoutBalwaysBsawBisntBmusicBoriginalBcomeBalmostBgetsBrightBseriesBmustBtheresBwholeBtimesBinterestingBroleBleastBguyBcomedyBactionBdoneBpointBfarBbitBscriptBamBminutesBmightBfeelBanythingBhesBsinceBprobablyBlastBfamilyBperformanceBtvBkindBworstByetBawayBfunBanyoneBeachBratherBsureBfoundBplayedBmakingBalthoughBgirlBourBbelieveBwomanBhavingBtryingBshowsB
especiallyBcourseBhardBcomesBputBgoesB
everythingBworthBdayB	differentBmainBmaybeBlookingBdvdBwasntBlooksBbookBendingBonceBplaceBwatchedBthreeB2BreasonBeffectsBsenseBsetBscreenBsomeoneBtrueBplaysBjobBduringBmoneyBtogetherBplayB10BsaidBeveryoneBinsteadBspecialBactorBseemBtakesBamericanBlaterBleftBseeingBaudienceB	beautifulBjohnB	excellentBhimselfBwarBversionBnightBideaBshotBhighBsimplyBfanB
completelyBhouseBniceBblackByoureBpoorBreadBusedBdeathBkidsBfriendsBshortBelseBwifeBhelpBalongBsecondBhomeBstarBmindBhalfByearBlessBtryBmenBgivenBneedBenjoyBuseBrestBboringBuntilB	recommendBeitherBtrulyBcoupleBperformancesB
productionBclassicBnextBstartBlineBdeadBwrongBstupidB	hollywoodBtellB
understandBletBfatherBcameBwomenBperhapsBgettingBterribleBfullBkeepBrememberBcameraBschoolBawfulB	wonderfulBmeanBsexBothersBmomentsBhumanBplayingBepisodeB
definitelyBnameBvideoBcouldntBitselfBbudgetBgivesBdoingBoftenBpersonBperfectBtopBsmallBearlyBfaceB
absolutelyBpieceBdialogueB	certainlyBguysBwentBfinallyBlinesBlikedBlovedBbecomeBtitleBfeltBcaseBheadBseveralBhopeBentireBsortBlostBstarsBwrittenBsupposedBoverallByesBstyleBmotherBworseB3BproblemBentertainingBtotallyBohBboyBsoundBliveBpictureBlaughBbasedBwasteBfriendBshesBseemedBagainstB	beginningBdarkBmrBcareBwantedBfinalBidBcinemaByoullBdespiteB	directionBbecomesBhumorBwontBfansBleadBguessBexampleBlivesBchildrenBalreadyBevilBunfortunatelyBcalledBgameB
throughoutBwantsBdramaBturnBlowB1BableBdaysBgirlsBunderBhorribleBwritingBfineBwhiteBqualityBtriesBtheyreBamazingBhistoryBenjoyedBkillBBmichaelBgaveBfavoriteBworksBkillerBsideBturnsBflickBbehindBexpectBactB	brilliantBpastBpartsBmatterB	obviouslyBsonBtownBeyesBrunBsaysBthinkingB	sometimesB
themselvesBstuffBdirectedBcarBmyselfBstartsBgroupBartBonesBsoonBheardBheartBviewerBtookBhighlyBkilledBdecentBhappensBcityBlateBgenreBactressBwouldntBfeelingBhellBkidBillBexceptBchildB	extremelyBcloseBlackBcannotBhourBwonderBfightBtoldBetcBleaveBokBmomentBparticularlyB	includingBsaveBstrongBdaughterBpoliceBshownBscoreBanywayBstoriesBlookedBhandBobviousBinvolvedBbloodBcomingBhappenB
experienceBitbrBviolenceBpleaseBcompleteBchanceBrolesBattemptBsimpleBlivingBtypeBstopB	hilariousBtakenBhappenedBvoiceBjamesBdavidBseriousBcoolBagoBrobertBopeningBusuallyBenglishBknownBletsBnumberBsongBsayingBmurderBexactlyB	seriouslyBcrapBacrossBjokesBslowBnoneBgodBhoursBhugeBwishBinterestBtalentBwhoseBmajorBorderByourselfBcinematographyB5BreleasedBrealityBcutBhitBpossibleBsadB	importantB
ridiculousBstartedBrelationshipBrunningBageBaloneB4BannoyingBheroBtakingBtodayBgoreBchangeBbrotherBfemaleBviewBshotsBmostlyBcallBknewBendsBsomewhatBpowerBbeyondBcareerB
apparentlyBsillyBbodyBturnedBwordBusualBopinionBdocumentaryB	basicallyBuponBhusbandBfindsBstrangeBknowsBhappyBscaryB	attentionBepisodesBwhatsBwordsBfourBtalkingBratingBarentBdueBlevelBclearlyBproblemsBmissBnovelBcountryBroomBlocalB	directorsBcheapBaddBbritishBsingleBmoviebrBjackBtellsBmusicalBdisappointedBtalkB
televisionBlightBsongsBeasilyBeventsBpredictableBmodernBsequenceBdialogBwhetherBappearsBreviewBlotsBfutureBbringB
supportingBsetsBhaventBneedsBfallsBsimilarBfiveBgivingBmovingBtriedBcertainB	enjoyableBtenB
soundtrackBromanticBwithinB	storylineBspaceBmentionBfrenchB	surprisedBaboveBparentsBshowingBentertainmentBhateBviewersBgeorgeBbunchBteamBdullBthemeBclearBcommentsBfilmedBelementsBtypicalBmiddleBstayBearthBfeelsBmessageBnearBfilmbrBcomicBamongBsorryBeffortBrichardBfallBthrillerBgreatestBworkingBkeptBwaysBusingBeasyBbroughtBreleaseBsequelBkingBbuyBsubjectBtaleBnearlyBnamedBsuspenseBsisterBwriterBmeansBdoubtBtheaterBfamousBgoneBeditingBmonsterBgeneralBactualBherselfB80sBdealBpeterB	fantasticB	realisticBimagineBclassBladyBstraightBavoidBrockBlearnBboysBsomehowBleadsBdrBmoveBformByouveBdieBreviewsBviewingBweakBpointsBbeginsBfeatureBforgetBhearBfastBrentBokayBmaterialBcheckB	sequencesBdeepB
particularBfigureB	animationBmysteryB
eventuallyBkillingBsurpriseBperiodBexpectedBsitBdogBwhosB	difficultB
believableB
atmosphereBwaitBcrimeBdecidedBtomBpaulBeyeBlameBforcedBdanceBshameBpremiseB	emotionalBindeedBbecameB	situationBaverageB	memorableByorkBscifiBseasonBtruthBredBpoorlyBcheesyBwhateverBsexualBneededBleavesBcrewBromanceBnoteBwriteBfootageBbeginBfollowBstandBpossiblyBkeepsBsuperbBmeetBunlessBreadingBwesternB
filmmakersBcreditsBminuteBoscarBthirdBsoundsBwritersBpersonalBmeetsBhandsBdoctorBnatureBtotalBfeaturesBtowardsBnorBopenB
interestedB	otherwiseBrealizeBsocietyBweirdBbattleBquestionBwhomBbeautyB
incrediblyBearlierBquicklyBmaleBboxBcopyBinsideBimdbBpreviousBjapaneseB
backgroundBeffectBcommentBgayBuniqueBplusB	directingBbadlyBhotBislandB	followingB	perfectlyBappearB
screenplayBamericaB20BcrazyBbringsBairBsettingBresultBolderBlaughsBbBfreeBmarkBworkedBvariousBfairlyBaskBadmitBstageBforwardBpowerfulBplentyBfrontBdevelopmentBmasterpieceBbillBstreetB	portrayedBapartBdumbBdeBspentBdeservesBwaterBattemptsBremakeBbabyBrichBpayBagreeBleeBdramaticBmeantBjoeBbusinessB	politicalBfireBactedBmessBideasBcaughtBcreepyBtwistBmatchB70sBcreateBbrothersBreturnBhardlyBwilliamBpresentBwastedBrecentlyBoutsideBcoverB
girlfriendBcleverBplainBreasonsBmanagesBpartyBlaBsuccessBlaughingBfailsB	potentialB	expectingBleadingBjokeBlargeBtalentedBdreamBcuteBcopBfightingBunlikeBbreakBmissingBoddBpureBsecretBpublicBholdBcauseBmembersBendedBtellingBpaceBslightlyBdisneyBcreatedBmissedBgunBwaitingBmarriedBcastingBcartoonBvillainBseesBsadlyB
incredibleBconsideringBvisualBescapeBbiggestBusesBproducedBfurtherBnudityBgermanBfitBlistBzombieBvanBdecidesB	portrayalBfantasyBfamiliarBamountBtrainBformerB
convincingBstateB	mentionedBcompanyBitalianBspeakBfollowsBentirelyBdiedB
appreciateBneitherBdancingBrateBpopularBwroteB12ByoungerBflatBeraBmovesBcreditBintelligentBspendBofficeBscienceBcomparedBtroubleBsuddenlyBpositiveB
successfulBvalueB	producersBcommonBchoiceBscottBkillsBviolentBtensionBforceBfilledB
situationsBboredB	christmasBaliveBsweetBlanguageB	audiencesBconceptBconsiderBcenturyBfocusBdecideBstoreB
ultimatelyBbasicBhairBfearBleavingBbizarreBimagesB	questionsBsickBlongerBexcitingBrecentBbandBsocialBrecommendedBmadBprojectB15BwerentBpatheticBsingingBmeaningBlikesB7BcoldBbooksB	pointlessBhonestBspiritBcontrolB
impossibleBawesomeB8BstudioBbarelyBdepthBcollegeBshowedBrespectB	involvingBchangedBamusingBvaluesBasideB
consideredBrevengeBhumourBgarbageBalienBwalkBaccentB	chemistryByeahBfairBboughtB	effectiveBfakeBcharlesBchannelBaddedBsoldiersBshootingBimmediatelyBgladBthanksBsolidBfailedB
impressiveBhonestlyBsurprisinglyBjaneBroadBthinksBmasterBadultBrunsBissuesB30B	somewhereBpickBarmyBtouchBstarringBgangBtwistsBbrainBsittingBnormalBbenBcampBgeniusBspoilersB	literallyBjimBwinBtermsB
conclusionBbotherBtrashB
personallyBsexyBblueBcultureBvampireButterlyBcomplexBwestBdrugBlikelyB
disturbingBaspectBprisonBfictionBabilityBstickBexplainBtoneB	adventureB	cinematicBcatBanimatedBtoughBsouthBmoodBcharmingBpurposeB	generallyBremainsBnobodyBsteveBsilentBjourneyBshootBwonBtripBweekBnakedBmanagedBlovelyB
appearanceBbeautifullyBparkBbedB100BplanBnaturalBindianBsceneryB	availableBchaseBsubtleBtasteBfrankBpicturesBgiantBchangesBphotographyBnowhereBknowingBcultBtouchingBlovesBemotionsBstandardBslasherBsupposeBcontainsB
constantlyBplanetBpiecesB
governmentBedgeBcatchBputsB
impressionBselfBbesidesBpresenceBpassBlondonBcomediesBmilitaryBcomputerBattackBappealBmakersB
historicalBfeelingsBdetailsBwalkingBjonesBplacesBchrisBbottomBsamBminorBmaryBthrownBslowlyBheyBwildBsmartB6BnoticeBinnocentBlandBsoulB	narrativeBharryBmagicBhopingBrideBnamesBequallyBdisappointingBcostumesBsurelyBdateBunbelievableB	everybodyBdoorBrareBoutstandingB	laughableBfilmingBterrificBdadByoudBexcuseBstandsBpainfulBoperaBghostB
adaptationB9BthusBfestivalBstunningBintendedBsentB	presentedBlawB	boyfriendBactsBbuildingBrayBcharlieBthrowBthankBcriticsBmakeupBfindingBaspectsBtiredBspeakingBmainlyBcentralBbriefBtrackBrandomBhurtBclimaxBzombiesBsupportBmistakeB
mysteriousBfullyBbruceB	detectiveBfinishBpainBhospitalBopportunityBmansBemotionBclubBsuggestBstudentsBraceBparisBimageBedBbondBaddsBwoodsBproducerBtonyBloudBlaughedBawardBgreenBcharmBmannerBcryBthemesBexpectationsBconfusedBchristopherBvictimsBofferBfreshBrussianBincludeBforeverBcopsBshipBpullB	confusingBwowBgradeBdreamsBvictimBspoilerBanswerBmotionBlivedBtwiceBaffairB
supposedlyBsummerBfunniestBfallingBfolksB	wonderingBmillionBlightingB
compellingBvsBlocationBhelpsBfellBnewsBjusticeBdrugsBbillyBplaneBmixB	exceptionBremindedBlacksB	seeminglyBcreativeBsixBrelationshipsB	impressedBdriveBextremeBdisappointmentBcontentBgorgeousBstudentBlikableBappearedBfascinatingBfacesBaheadBratedBheavyBtimebrBflicksBbornBbecomingBagentBsmithBcaptainBallowedBpaidBdiesB90B	developedBdeliversBbatmanB	thereforeB	happeningBapproachBchurchBpickedBholesBmoralBprovidesBoffersBnegativeBloverBintenseBgemBcolorBadultsBmarriageB
differenceBbarBdrawnBcompareBcgiBshockBreturnsBelementBshareBworthyBstuckBmerelyBmartinBdetailBsuicideBbossB
attractiveBrentedBputtingBinformationBgroundBfollowedBclichéBuglyBteenBserialBshouldntBcountBmsBmomBradioBontoBanimalsBhasntB	forgottenBflawsBeventB
collectionBshockingBpornBkeyBindustryBtrailerBliesBsystemBquickBgraceBwoodenBtortureBledBiiBdisasterBdirtyB	standardsBkevinBimpactBbeatBareaBangryBspotBmediocreBabsoluteBsuperBreadyBmovedB	christianBstewartB	americansBsearchBmachineBnastyBimaginationBpersonalityBlistenBhotelBseaBmemberBteenageBhelpedBwearingBturningBflyingBdirectBdeliverBcarryBtragedyBepicB
thoroughlyBloseBcreatureBaskedBthomasBtragicBphoneBkellyBheldB	actressesBafraidBstatesBsoldierBlatterBdonBdamnBartisticBsevenBprovesBinspiredBhenryBactionsBhiddenBgamesB
filmmakingBcontinueBanymoreBstationBrealizedBprovideBenergyB
whatsoeverBjudgeBwantingBmonstersBdyingB
commentaryBtimBwillingBteacherBmurdersBfellowB	redeemingBprocessBpopBadditionBtodaysBprofessionalB	favouriteBstruggleBfashionBallowBwilliamsBfoodBdeserveBbitsBbeganBwallBtrustBsecondsBrarelyBjasonB	dangerousBcallsBjerryBacceptBpassionB
intriguingBasksBqueenBextraBintelligenceBclichésB	childhoodBnumbersBdescribeBtheatreBpleasureBunderstandingBsuperiorB	necessaryBclothesBbloodyB	apartmentBlimitedBindependentBholdsBdesignBdeeplyBtearsBsuspectB	filmmakerBphysicalBdoubleBanywhereBwarsB	scientistBhorseBzeroBfoxBapparentB
surprisingBsucksB
introducedBcomedicBgoldBfatBstephenBstepBaccidentBgrandBanybodyBsatBmikeBlearnedBchineseBwonderfullyBsightBremindsBmemoryBieB	watchableBwarningBryanBincludesB
friendshipBartsBwhilstBnoirBblameBthinBsomebodyBmouthBmoonB	desperateBaccurateBringBmartialBdesertBunusualBmonthsBdannyBbuildBbrutalBmentalBhatedBdrivingBalanBrapeBexplanationBlordB	locationsB60sBplotsBheroesBunknownBacademyBsleepBuncleBjrBmineBheadsBbrownBanneBstockBloversBabsurdBvillainsBnicelyBjacksonBthembrBjoyBhadntBfaithBanimalBplayersBmemoriesBjohnnyBgagsBscaredBpacingBengagingBdiscoverBcapturedBwindBvisionB
remarkableB	religiousBlooseBkeepingBalBoppositeBnoticedBhumansBcriminalBcreatingBmrsBbrightBplayerBleaderBconstantBboatBbelowBtreatBrecordBcarsBnormallyB	knowledgeBgenuineB	communityBskipBjeanBbrianBstartingBsingerBpowersBheresB
generationBboardBsistersBrobinBoccasionallyBarthurBvhsBresponsibleBissueBfordBwoodBproveBnumerousBmetB	lowbudgetBkillersB	accordingB	technicalB	naturallyBincludedBhitsBfloorBsavingBrelateBbiggerBawareBsmileBseanBmissionBlackingBartistB50BmanageBgaryBfieldBdeservedBcaptureBblandBgrowingB
connectionB
referencesBpilotBmilesBlegendBfightsBcrappyBmagnificentBhopesBunnecessaryBterriblyBlovingBjeffBeddieBtreatedBordinaryBfinestBaliensBpriceBpulledBnationalBjenniferBcurrentBwindowBdealingBconflictB
originallyBhumorousBgoldenBemptyBdealsBwittyBlossBjoanBeuropeanBeatBbehaviorBunfunnyBdavisBregularBprivateBpairBmurderedBfranklyB	featuringBeffortsBbobBofficerBlengthBjumpB	genuinelyBforeignB	explainedBspanishBrollBwitchBweveBhigherBcrossBsoapBmorningBforcesBfailBdesireBblindBnonsenseB	meanwhileBhumanityBgottenBfinishedBessentiallyBdressedBcutsBanthonyBrealismBquietBpsychologicalB	hitchcockB	concernedBcB	reviewersBpriestBmixedBfameBthisbrBmediaBtapeBsingBexistButterBrubbishBrevealedBtowardB	teenagersBsuckedBstoneBpartnerBfavorBenglandBwheneverBstevenBshopBprogramBpretentiousBpageBnickBcableBvillageB
underratedBsignBhowardBclassicsBbrokenB40B	nightmareBdatedBbuddyBbreaksBstreetsBsheerBkindaB
discoveredBallenBtwentyBgunsBtexasBnativeB
comparisonBawkwardBadamBstealBownerBcrowdBunableBtinyBfeetByouthBvampiresBshallowBsatireBluckyBhunterBgrowBcastleBvisitB	presidentBattitudeB1010BstereotypesB	screamingBportrayBmaxBfateBdrunkBdebutBstudyBjimmyBheroineBgraphicBevidenceBmooreBcornyBadviceBwideBscreamBsavedB
rememberedBreceivedBprotagonistBinternationalB
flashbacksBeditedB	continuesBweddingBinstanceBfailureBworldsBtrekBskillsBriverBresultsBranBbringingBballB
unexpectedBpeoplesBinsaneB	creaturesBandyBiceBvisualsBtalentsBstrengthBreachB
delightfulBdogsBbombBvoteBsurviveBreactionB	deliveredB
australianBgoryBgonnaBfootballBcontextB1950sBspectacularBlogicBfillBfaultBdecisionB	contrivedBcombinationBopensBlessonBkickBallbrBmeetingBladiesBjulieBericB	describedBpostB
irritatingBfredBdreadfulBteenagerBtaylorBsuitBsellBcameoBprovedBjesusBgangsterBdickBultimateBidentityBemotionallyBbarbaraBvisuallyBtravelBparodyBonebrBbrilliantlyBwinningBshakespeareBprovidedBluckBhimbrBcreatesBcleanBunrealisticBtalksBinvolvesBheckB	discoversBdanB
commercialBbeachBrangeBheavenBdBsheriffBprinceBakaBrecallB	halloweenBcontrastBbelievesBremainBgrewBfranceBenemyBdragBcausedBauthorBasianBwiseBpromiseB	hopefullyBfathersBcapableBtheyveBstandingBreliefBperspectiveBlosesBframeBfinaleBcuriousBcostBseatBfreedomBeatingBannBflyBalexBoverlyBmebrBgeneBassumeBsurrealBstealsBsiteBrescueBlewisBlevelsBformulaBaskingB	existenceBendlessBdouglasBdecadeBaliceBwellbrBtraditionalBstronglyBsendBmattBfactsBdestroyBbankB	treatmentBsakeB
disgustingBdevelopBcashBcandyBproductBnorthBeuropeBcenterBcaresBweeksBtheatersBreviewerB
individualBcrashBchickBtypesBproperBholmesBancientBagesBvoicesBspeechBhallBbodiesBsympatheticBchooseBallowsB50sBviewedBtwistedBthatbrB	subtitlesBsequelsBroundBresearchBportraysBneverthelessBjapanBthoughtsBrobBlaughterBkongBexecutedBmajorityBinsultBbuiltBannaBstudiosBstoppedBspeedB
portrayingBpleasantBplansBlouisBawardsB
propagandaBwalkedBrulesBsurfaceBsuddenBlousyBlargelyBbetBtillBlynchBharrisBshockedBlearnsBlakeBclueB
amateurishB	universalBhandsomeB
gratuitousBextrasBchiefBvehicleBukBterrorBcircumstancesB1970sB	virtuallyBhorriblyBguiltyBfactorBexperiencedBdollarsBcartoonsBtestBspoilBskyBmodelBkeatonB	committedB
theatricalBsoftB	slapstickBpityB	painfullyBexcitedBdennisBcorrectBunitedB
technologyB	sufferingBproduceBhauntingBembarrassingBwalksBroseBpassedBlosingBholdingBentertainedBdriverB90sBstorybrBpatrickBnetworkBlesbianBdepictedBcapturesBrentalBfitsBcoreBteensBserviceB
relativelyBmarryBhuntB
depressingBtarzanBtableBdubbedBtendBmattersBfamiliesBassBunfortunateBpracticallyBlatestBbearBsourceBrobertsBreligionBrBmorganBdarknessBconvinceBchosenBblondeBwayneBvarietyBsaturdayB	recognizeB	influenceBtediousBgermanyBexploitationBdevilBwinnerB
satisfyingBweekendBspendsBriseBfBenterB810BwitBsportsB	qualitiesB	performedBpatientBpaperBdrawB	daughtersBcageB	appealingBhideBcostumeBclaimBdisplayBcontemporaryBasleepBangelBtrainingB	substanceBskinBhearingBhalfwayBextraordinaryBdangerBspeaksBsegmentBpresentsBjungleBclichédBappropriateBgrownBfiguredBexperiencesBblowBwerewolfBmindsBjackieB	childrensBcanadianBbmovieB11BtedBsympathyBsharpBseasonsB
previouslyBlarryBdryBcryingBcallingBwitnessBscareBharshBdeadlyBwesternsBvietnamBseagalBruinedBrealizesBidiotBhangingBhandledB	favoritesBdirectlyBconversationBamateurBsuffersBscaleBmovementBexpressBbreakingBanglesB
adventuresBturkeyBtrilogyB	offensiveBmountainBversionsBmaskBkateBinitialBedwardBuniverseBservesBjosephBfeaturedB	elizabethBdegreeBdeanBclaimsBaccidentallyB	promisingBhitlerBhauntedBfleshB
departmentB	encounterBdonaldBcostsB
continuityBwarmBveteranBsecurityBsafeBprimeBhillBgordonBfridayBeveningBchoseBalbertBsuspensefulB	statementBobsessedB	nominatedBcloserB	surprisesB
refreshingBpriorBkissBgrantedBexactBdudeB25B13BwalterBsupernaturalBpsychoB
occasionalB
excitementBdropBcuttingBbelievedBbbcBandorBaccentsBtruckBsirBforgotB710B	structureBripBnurseBcoveredBbroadwayB
worthwhileBrogerBremotelyBmassiveBgreaterBforthBfootBcategoryBbotheredB	amazinglyBviaB
mainstreamBlightsBlaneBjuliaBeightBchinaB	abandonedBundergroundBsouthernB	professorB
interviewsBcruelB
californiaBwouldveBweaponsB
thankfullyBroyBrevealBrequiredB
reputationBinsightBwearBtargetBsimonBpreferBpacedB
overthetopBmarketBlauraBirishB	interviewBruleBregretBpullsBlowerB	executionBeroticB	destroyedBbrooksBbettyBsunBstorytellingBstereotypicalBpoliticsBcellBalrightBaintBsoundedBsectionBruinBproudBpreparedBjBgrittyBfrighteningBfishBwhoeverBwaybrBstayedBrussellB	regardingBparkerBmagicalBhiredBeffectivelyBdemonsBabuseBstolenBrobotBpositionBplacedBpayingBherebrBgrantBexplainsBwellesB
washingtonBrightsBraisedB	narrationBholeBforgettableBdeathsBsupermanBmultipleBmistakesBisbrBheavilyBendbrBcolorsBunlikelyBreporterBproductionsBmickeyBgraveBfocusedBfbiBdeeperBcriticalBanimeBtouchesBsortsBprincessBpraiseB	listeningBjunkBjewishB	highlightBflightBbusBartistsBwinsBtrappedBskillBserveB	melodramaBindiaBfalseBdemonBblockbusterBbaseballBaccountBsummaryBsuckBstrangerBscenarioBpeaceBofferedB	initiallyBhatB
everywhereBdrivenBdraculaBdecadesB	convincedBurbanBtightBtheoryBstaysBprintBkindsBchaplinBcampyBbaseBarmsBwilsonBsundayBsidesBpileBmatureBkimBhedBfoolBfacialBbuyingBwelcomeBteethBsonsBsBrevealsBrentingBnaiveBforestB	downrightBcarriesBripoffBquirkyBlifetimeBfareBcurseBblownBarrivesBuBthrowingBspyBspiteBnuclearBkaneB	depictionBtaskBpathBoutbrBnightsBenvironmentB	criticismBamazedBafricaB410B1980sBvictorBusaBthrowsBrepeatedBgoodbrBflowB	flashbackB
expressionBdressBafricanB2001BtouchedBsubplotBracismBmexicanBjailBdanielBbeastBwriterdirectorB
understoodBtechnicallyBmildlyB	legendaryBfocusesBexamplesBdrewBdesignedBcomplicatedBcivilBchoicesBuninterestingBrainBmurdererBlukeBinnerBandersonBunconvincingBroutineBmereBgoofyBcouldveBviewsBsusanBstBspoofBsitcomBshortsB	sensitiveBrollingBoriginalityBlearningBimageryBtermBpaintBhandleBfourthBfiguresBcombinedBbirthBagainbrBwarnedBsignificantBseekBscaresBclosingB	atrociousBstatusBpassingBmothersBeastBdrivesBaddingBreminiscentBlifebrBguestBdozenBdinnerBculturalBbraveBbeliefBpurelyBperformBintellectualBcrudeBchangingBblahBwoodyBtimingB	screeningBpregnantBonbrBnecessarilyB
miniseriesBexpressionsBdeviceB	bollywoodBangerBachieveBremoteBnudeBignoreBhongBhelpingBcodeBchargeBcarriedBbugsBtreasureBtitlesB
scientistsBsarahBroughBjohnsonBinspirationBgasBfunnierB	disbeliefB
attemptingBangleBproperlyBmetalBjobsBconcertBcausesB	afternoonBwearsBringsBrawBracistBindieBdollarBtalesBsleepingB
regardlessBlonelyBlieBintroductionBhelenBeverydayBdesperatelyBdescriptionBchuckBboreBservedBseparateBratingsBquestBprotagonistsBmillerBlawyerBjoinBfaithfulB	entertainBsufferBronBgruesomeBprotectBinternetBflawedBdeliveryBattacksBamongstBscriptsB	referenceBnonethelessBmassacreBlettingBdrinkingBchanB110BtommyBoliverB	stupidityBpicksB	ludicrousBlosBletterBenjoyingBdailyBoBnovelsBmurphyBdrinkBcontroversialBcontainBcontactBcarryingBbasisBarmBangelsBwhereasBupsetBshutBretardedBmgmBlisaBfuBeastwoodBappreciatedB	thousandsBineptBhostBcredibilityBbreathtakingBanswersB910BwolfBtopicBsucceedsB	strangelyBsantaBnotableBlazyBformatBfabulousBcomicalBbelovedBsumBshadowBrevolvesBparBnonexistentBhenceB
determinedBbarryB
strugglingBpoignantBmadnessBentryBcynicalBcomedianBcaringBbreathBwealthyBsleazyB	obnoxiousBguessingBghostsBfallenBalbeitBwwiiBstepsBmentallyBflawBexceptionalBegBcredibleBcousinBsoldBsinatraBmobBguardBglassB	fictionalB	countriesBchillingBblairBxBtroubledBmedicalBleB	catherineB	authenticBacceptedB	traditionBrelatedBnowadaysBleagueBjayBinterpretationBheartsBgraphicsBexistsBembarrassedBcourtBbobbyBvacationBtreeBtenseB
revolutionBpunchBopenedBkarenBhoodBfairyBclarkB310BunionB	strugglesBmindlessBjonBdareBcooperBcheeseBashamedBappearancesBslightBsingsBsettingsBitalyBidioticBgreatlyB
frequentlyBescapesBconsistsBbradBbagBworryBwarnBstomachBstealingBstarredBproofBlessonsBfortunatelyBfarmBdearB	challengeB	australiaBsocalledBsinisterBscreenwriterBmarieBjessicaB
experimentBdaveBwarnerBsuggestsBsnowBrushBjamieBdislikeBconveyBswordBsanBregardBgodsBdragonBbalanceBtoobrBthousandBstormB	sexualityBseekingBreplacedBhardyB	happinessBextentBchairBcasesBbirthdayB	assistantB	advantageB1930sBweaponB
meaningfulBlovableBhorrificBexpertBvincentB	techniqueBsufferedBmichelleB	invisibleBhundredBgrippingBclipsBattackedB45BtuneBstopsBsovietBrippedBplasticB
physicallyBmillionsBironicBguideBgrossBsallyBovercomeB	marvelousBcowboyBbadbrB2005BvonBsolveBriskBmusicalsBmst3kBleslieB	intensityBhidingB	criminalsBcraftedBconsequencesBcolumboBchasingBvideosBstylishBstuntsBsendsBremindBrefusesBraiseB	obsessionBnazisBlegsBjumpsB	innocenceBhoffmanB	franchiseBalasB80BthiefBtextBspokenBshinesB	searchingBimaginativeBfriendlyBdelightBburnsBandrewsBwalkerBstruckBpresentationBoddlyBknifeBhintB	conditionBairedBtrapBtiedBrepeatBnBjonathanBhaBdealtBbitterB	appearingBwastingBtributeBtrialBnotedB
intentionsBdawnBcourageB	attemptedB14BuselessBstanwyckBridingBouterBobjectBnonBhonorBdramasBwomansBtrickB	thrillingBspendingBsilverBninjaBneedlessBmatrixB	instantlyBhorsesB	confusionBcolorfulBbelongsBsuccessfullyBshootsBpsB
perfectionBmastersBlockedBironyBhundredsBfacedBdisagreeB	cardboardBcameronBtourB
performersBnoseBmexicoB	kidnappedBhangBcheckingBbrokeByoubrBsurroundingBreynoldsBquoteBlloydBlackedBholidayBgottaBfreemanBequalBcreationBcornerB	connectedBcloselyBspecificBsilenceBshowerBmanagerBkungBkicksBidentifyBfxBcrimesBclintBburtBauntB
attractionBachievedBtornB	thrillersB
repeatedlyBpatBpacinoBlesserBhillsBglimpseBdraggedBrachelBpackBgreekB	elsewhereBboundBbladeBtherebrB
sutherlandBsucceedBshedBsavesBrushedBportraitBmariaBlyingBhusbandsBheistBgrimBflashBessenceBchasesBbucksBbasementBunforgettableBtoyBtalkedBstringBoilBnaziBmidnightBkennedyBcaineBunintentionallyB	typicallyBtheydBspotsBrivalBprofoundBnotchBnavyBhealthBhappilyBgagBdevelopsBcureBannoyedB1960sBuncomfortableBteachBsuitsBshallBmiscastBmirrorBindiansBhboBgutsBgermansB	competentB	carefullyBbridgeBsubBshoesBrowB	revealingBralphBpoolBpitchBlibraryBbusyBblowsB	ambitiousBwannaBsandraB	ourselvesBnineBindividualsBensembleB20thB
uninspiredBshortlyBshapeBpetBhookedBestablishedBbrieflyBbeingsBbeerB18BsoulsBsnakesBsnakeBshadowsB
revelationBmelBfreddyBeasierBdianeBdialogsBbrandBarnoldB1stBtonsBsentimentalBsellingBrogersBrocksBpointedBpersonalitiesB
outrageousBmoviesbrB	essentialBcompetitionBbibleBadBsophisticatedBsharkBrelevantBincreasinglyBimportantlyBgloryBexerciseBdvdsBdoorsB	curiosityBcatholicB
acceptableBtriteB
techniquesBtallBpulpBpianoB
performingBgBboredomB	appallingBamandaBstretchBstanleyBsizeBsinBridB	remainingBpullingBprideBposterBoscarsBopposedBmeatBhandfulBguiltBgrowsB
encountersBemilyBconcernsBcatsBagedBtricksBstoleBloneBhomageBhardcoreBfifteenBdroppedBdragsBatmosphericB
associatedBangelesB	alexanderB	wrestlingBsubplotsBsidneyBpowellBnobleBneatBmouseBjumpingB
cameraworkBadaptedB2006B2000BwakeBtravelsBthrewBthirtyBspoiledBscarlettBresponseBpersonsBpackedBmelodramaticBlaidBinteractionB	inspectorBgothicBgoalBdefiniteBcameosBbatBstrikingBsetupBsegmentsB	returningBpittB	onelinersBneighborhoodBmassBkudosBkoreanBjerkBhorridBbackdropBzoneBweightBwatchesBpushB	providingBoceanBjakeBintentBincidentBhopedBfancyB	enjoymentBwondersBwaveBwatersB
surroundedBstuntB	spiritualBsexuallyBperBmonkeyBmagazineBloserB
ironicallyBinfamousB
incoherentB	godfatherBdoctorsBcreatorsBcaveBcabinBburningBbetterbrBargueBadmireB	possessedB	notoriousBlooselyBkillingsBeveBdistantBdancerBcharismaBbearsBbattlesB
admittedlyBstoodBozBoccursBminimalB
horrendousB	endearingB	discoveryBdestructionBdemandsB	carpenterBburtonB	broadcastBboneBbirdB40sBshyB
redemptionBofficersBluckilyBloadBlikingBfosterBdubbingBcorruptBcoachBcluesBaccusedBvagueBthrillsBspecificallyBreallifeBpuppetBnationB	miserablyBidiotsBhollyBherbrBgainBflawlessBfearsB	expensiveBexBclaireBcharacterizationBcardBbeatingB	attractedBwishesBwickedB	territoryBstrongerBpreciousBjeremyB	intentionB
importanceBforgiveBengagedBbuckBwillisBtorturedBrequiresBrealiseBrapedB
presumablyBpieBoutcomeBnoiseB
motivationBheatBfunnybrBfilmsbrBdevoidB	countlessB
conspiracyBallowingBairportBtheyllBsuspectsBsuperblyBphotographedBneighborBmansionBlawrenceBkhanBignoredB
highlightsBhalBfortuneBflynnB
explosionsBemmaBeerieB	dinosaursBcommitBcomfortableBcolourBchicagoBbeatsBarriveBarrestedBaforementionedBwallsBtoiletBstiffBstatedBpressBnormanBhamletBelderlyBdisplaysBdigBcorpseBchancesBbuildsBswedishBsurvivalB
subsequentBstrikesBstanBspareBsomeonesBsmokingBranksBphilipBlionBhypeBhuntingBentersBcausingBbrazilB30sB0BupbrB
tremendousBthreatBtBspookyBsavageB
restaurantB
repetitiveBpickingBmonthBgroupsBglennBfighterBdirectorialBbergmanBbeatenBbareBalltimeBagentsB	abilitiesBvastBtradeBstayingBstaffBmidBmessagesBlindaBkenB
hystericalBhudsonB
frustratedBescapedBdetailedBcouplesBcloseupsBaudioB	symbolismBsplitBscriptedBrickB	resemblesB
reasonableBpossibilityBmitchellBlBgrayB	everyonesBdistanceBcommercialsBchaosBburnBbeneathB	believingBareasBwonderedBthumbsBsecondlyBruralB
reasonablyB	reactionsB	overratedBneckBnearbyBmarksBhorrorsBheroicBfulciBfittingBdocumentariesB
discussionB	continuedB
concerningB	complaintB
cinderellaBciaBcharismaticBaccomplishedBtitanicBstrictlyBstoogesBsmokeBpleasedB
pleasantlyBplagueBofferingBmatthewBitllBirelandBhipBfrancisBdigitalB	celluloidBattachedB
altogetherBrootBreturnedB	prisonersBplanningBnancyBmustseeBmakerBlugosiBlucyBlaurelB
inevitableBianBhireBhBgiftBgenresBexploreBemBcupBcrisisBcombatBcoleBchristBburiedBbroadBbreastsBanywaysBanticsBandrewBsubjectsBrussiaBreceiveBpinkBpersonaBmediumBlogicalBfactoryBeditorBconversationsBburnedBbritainBairplaneBaffectedB17BvirginBvalleyBtunesBridiculouslyB
resolutionBpreventB
overactingBnelsonB	murderousBmildBiiiBhittingBfarceBdrawsB
deliveringB	currentlyBcraigBcaredB
afterwardsBvictoriaBreachedBoverdoneBnotablyB	landscapeBinvestigateBhanksBhammerBdoomedBdaringBcriedBcontraryBcitizenBcarolBaimedBswearB
overlookedBobscureBmileBhatredB	executiveBevidentBdutyBdollBbrideBbellBunwatchableBunbelievablyBtrioBsticksB
simplisticB	secretaryBremovedBoccurBmarthaBharderBfreakBfondaBextendedBdrunkenBcruiseBcookBchainBbleakBadamsBwardBtwinBticketB
spoilersbrBsidekickBsecretsB
representsBrankBpreviewBpovertyBmeaninglessBlargerBinvolvementB	inspiringBgenerousBfatalBcoversBcomicsBcampbellBbrainsBwallaceBupsBundoubtedlyB	travelingBsadisticBpushedBprojectsBnotesBmurrayBmeritBmarioBkurtBjuvenileBjazzBidealBgrandfatherB	graduallyBgenericBfurthermoreB	dedicatedB	classicalBwalkenBstripBsignsB
scientificBromanBlistedBkitchenBjealousBimplausibleBfuneralB	elaborateB
disjointedBcriticBconnectBwreckBwindsBvirusBtoddBtitledBtimelessBspinBspikeBoffendedBmoralityBinvestigationBfailingBetB
depressionBblendB24BworriedBwinterBwarriorBwannabeBselfishBreliesB	lifestyleBgreyBfingersBdiseaseB	dialoguesBclumsyBcarlBcamerasB
brillianceBwBshipsBseenbrBrageBproceedingsBpredatorBpornoB	insultingBconventionalBcomplainBbluesBbathroomBangelaBviciousB
terrifyingBscreamsBsadnessB
recognizedB
prostituteBprisonerB	primarilyB	newspaperBmissesBmenacingBlyricsBironBimprovedBexplicitBdireBdiamondB	buildingsBargumentB2ndB16BworkbrBthreateningBrandyBpopcornBpoliticallyBneilBinstallmentBgrabBgoodnessBfingerBdevotedB	describesBconBcareersBcanadaBbuddiesBbandsBadviseBachievementB1999BupperB
unpleasantBshiningBreducedB
populationBopinionsBmatthauBlightheartedBisolatedBestateBelvisBdorothyBbuttBblockBbinB	worthlessBwebBtreesBtracksBsubtletyBstagedBsportBsoleB	slightestBofficialB
influencedBhighestBfrankensteinBdustBdifferencesBcrackBbootBbakerB	yesterdayBturnerBthroatB	tarantinoBreunionBpunkB	psychoticB	producingBholyB	eccentricBconneryBcoffeeBbuttonBarrogantBalikeB1940sBwaxB
universityBstylesB
stereotypeB	representBrecordedBreachesBprimaryBpartlyB	movementsB	mountainsBmBjustinBjudgingBgentleBfondBduoBcraftBcatchesBbushBbatesBannieBvegasBunintentionalBteaBshowbrB	recordingBpurchaseBplantBpaintingBmadonnaBjustifyB	inventiveBerrorsBenormousBeBcharactersbrBcarterBcardsBwrappedBspellBsloppyBsatanBpsychiatristB	provokingB	principalBpretendBordersBnotbrBlouBintrigueBillegalBhopperBfrancoBfiredBfieldsBdaviesBconsistentlyBblewBbenefitBtranslationBtaughtBsoloBshellBpearlBpantsBmixtureBloadsBlabBkiddingBhousesBexposedBdukeB
developingBdentistBbourneBblatantBagingB210BtrailBtracyBthickBtearBsuperficialBstrikeBstevensB	renditionBpropsBnamelyBmafiaBloyalBkingsBkickedBinbrBhookB
hitchcocksB	explosionBdignityBcringeB
convolutedBcontractBcheatingBabsenceBwatchbrBversusB	strangersB	similarlyBrabbitBplannedB
philosophyBowenBoughtBmistakenBmenaceB	manhattanBmaintainBkirkB	immenselyBheadedBformsB
explainingB	educationB	disturbedBdeliberatelyBdamageBcomposedB	comparingBbelongBamyBamountsBalternativeBalfredBagreesB911B2004ByaBtrailersBsolelyBsmallerBrobberyBrelativeBpotentiallyBpoetryBpennBoccurredBmorebrBmethodBliberalB	intriguedB
innovativeBimaginedBhuhBfliesBempireBdepictsB
consistentBcalmBwritesBworeBuserB
terroristsBsplendidBspiritsBrisingBpassesBoddsBmontageBmodelsBmeasureBjuniorBinvolveBhonestyBhistoricallyBfocusingBexploredBdrivelBdistractingBcolonelBcaptivatingBbottleBagreedBworkersBwetB
unoriginalBtransformationBslickBsentenceB	sacrificeBruinsBrecognitionBrangerBracialBphotographerB	performerBmuseumB	miserableBmargaretBinstantBheartwarmingBexistedBevaBdocBdiscussB	depressedBdarkerB	commentedBchoreographyBbrunoBborderBboldBadvanceByellowBthrillB	terroristBpushingBhatesBdinosaurBcliffBbulletsBassassinBahB	witnessedBtankBnedBmummyB
mentioningBguestsBgrandmotherBgialloBgardenBexaggeratedBeatenBdiscBconstructedBclosetBaffordBwingB
thoughtfulBstellarBsitsBsimilaritiesBscoresBpatientsB
passionateBmateB	masterfulB
introducesB
homosexualBglassesBendingsBemphasisBdatingBdaBcombineBclosedBchannelsBbonusBbirdsBarmedBunfoldsB
unbearableBterryBtenderBsoccerBreportB	plausibleBnarratorBmyersBlinkBleonardBjoinedBfloatingBexchangeB	equipmentBeightiesBedgarBcountrysideBcoherentBchasedBcarefulBblondBbetteBworldbrBtagBswimmingB	succeededB	satisfiedBritaBrapBpuppetsBproceedsBpracticeBnotionBintimateB
hollywoodsBhollowB
helicopterB	gangstersBfrostBfixBetcbrBcrucialBchestBborrowedBblankBadorableBurgeBsynopsisBsuitedBsquadBshirtBscenebrBruthBrivetingBpoeticBparallelBoverwhelmingBmatchesBlucasBlayBhilariouslyBgloriousB
futuristicBfrankieBfloridaBcurtisBconsiderableBcircleBcarreyBtempleBteachingBtameB	survivingBstraightforwardBstiltedBspringBshineBroomsBrevolutionaryB	relationsBpromisesBpaysBonscreenB	neighborsBearnedBdozensBclothingBchildishB	capturingBborisB	alongsideByellingBunderstatedB
underlyingBtongueBtiesBstreepBschemeBsandlerBruthlessBresemblanceB
reflectionBreactBphilosophicalBnicoleBmotivesBimpressBhurtsBhughBfrequentBfoughtBfooledB
enterpriseBemperorBcontestB	companionBbiteB60BwebsiteBtrainedBtieBstaringBshelfBscreenbrB	scarecrowBpartiesBparentBpaintedBingredientsBimproveBillnessBdivorceBcusackBcopiesBclosestBbulletBbuffBbrandoBblakeBaidBadequateBaccompaniedBvividBvaluableBshirleyBscenesbrBreedB
punishmentBpopsBplotbrBpBoccasionB	laughablyBguessedB	guaranteeB	franciscoBegoB
disappointB	decisionsBcoxB	containedBarrivedB3rdB3dBwishingBwineBunevenBtwelveBtonightB	suggestedBsteelBsmoothBquotesBpromisedBportionBpetersBlipsBharveyBdougBderekB
corruptionBclownBcleverlyB	charlotteB1980BwishedBwasbrBtendsBsuitableBresortBquinnBmorrisBlimitsBjeffreyB	illogicalBharoldBexoticBdropsBdisappearedBdirectsBcommandBcdBbusterBassaultBusefulBtreatsBtoplessB	survivorsBslapBreflectBnooneBmoodyBmayorBlegBknockBkeithBinappropriateBheartbreakingBgrudgeB	financialBendureBeditionBeasternBcannibalBblowingBbeliefsBalbumB70B35B3000B	vengeanceBtrashyBtadBswitchBsquareBrubberBresponsibilityBphraseBmoreoverBincompetentBinaneB
horrifyingBhideousBhandedBenjoysBdynamicBdammeBcrystalBcarrieBcagneyBbrutallyBboBadvertisingBwoundedBwidowBtwinsBsurvivedBstinkerBshakeBremainedB	purchasedB
pretendingBpossibilitiesBphilBonedimensionalB
nominationB
nightmaresBmankindBknightBjoshBenemiesBdrawingBdownhillB	displayedBdescentBcheckedBcheBbridgesBblacksBawakeBamericasBabcB2002BweakestBspokeBsincereBsafetyBrottenBrocketB	relativesB	realizingB	nostalgiaBnonstopB	marketingBmarionB	madefortvBlincolnBjawsB	forbiddenBfeverB
creativityB
conditionsB	carradineBbebrBbannedBabysmalBaboutbrB2007B1995B	wanderingBwackyBtransferBrunnerBprogressBperformsBmacyBlolB
journalistBjoelBhartBgrahamBfolkBflopBelBeitherbrBdutchBbumblingBaliB
widescreenB	voiceoverBunderstandableBtouristBthugsBthruBrolledB
remarkablyBreadsBphantomBonlineBmodeBminimumBminiBlustBlegalBhuntersBglobalBelegantBeducationalBdemandBconvincinglyBcheatedB1972BwesBwealthB	superheroB	resultingBnervousBmistressBgraspBgarfieldBfrustratingBdiscoveringBdancersB	commanderBcoastBclaudeBbackgroundsB	alcoholicBwomensBwatsonBvirginiaBunitBtrafficBteachersBroyalBrepresentedB
progressesBnycBjuneBjudyBhomelessBfacingBengageBembarrassmentBeaseB	disappearBdimensionalBcrushBclassesBclaimedBcinematographerBchickenBbehaveB	authorityB	attitudesB
vulnerableBveniceBturkishBtimesbrBseeksBrelationBpitifulBpackageB	mysteriesBmarchBlatelyBguardsBfrustrationBeagerBdestinyB	depictingBcolinBchapterBchallengingB
carpentersBbuffsBbranaghBbedroomBbarneyBbangBbaldwinBavoidedB
artificialB2003B1990sBwithbrBwisdomBwarmthBwaitedBspreadB	seventiesBsensesBscottishBsappyBrockyBpanicBmotivationsBlitBjetBhintsB	greatnessB	formulaicBfelixBfedBfasterB	expressedBexceptionallyBelephantBelectricBeatsBdollsBdesperationBdesiredBdancesB	communistBcitizensBcgBcannonBbollBarrivalBappreciationBwoundBtriumphBsolutionBrobotsBquitBpropertyBlikewiseBinvasionBincomprehensibleBignorantBhamiltonB	grotesqueBgiftedBearsBdefenseBaustinBarguablyB	alternateBadaptationsBtylerBtwilightBsullivanBsueBspainBsoftcoreBshowcaseBsevereBrobinsonBprepareBpremiereBphonyBpeoplebrBpatienceBmtvBmassesBmarshallBmaniacBleighB
irrelevantBgusB
fascinatedBfamilysBernestBdefeatB	conceivedB	climacticBcinemasB
cassavetesBargentoBaaronBwellsB	wellknownBwavesB
suspiciousBstandoutBsmilingBscopeBrebelBprizeBnorrisBmightyB
lonelinessBloadedBlastedBlandsBkingdomBheadingBgenerationsB
enthusiasmBcomposerBcivilizationBbernardBaprilBapesB	affectionBaffectBwildlyB	trademarkBtiresomeB	spielbergBsignedBresistBpressureBnutsBmonkeysBkarateBfaultsBdamnedBconservativeBcircusBcatchyBbullBboxingB	behaviourB1996BwarrenBvaguelyB
underwaterBtrackingBtownsBtimothyBstressB	spaghettiBspaceyBpettyBmuddledBmaggieBlowestBkennethBintentionallyB	householdBgloverBellenBdrearyBcostarB	correctlyBconstructionB	comediansBawhileBastonishingBadvancedB
accuratelyBwivesBwedBtroublesBtromaBstrandedBstarkBsplatterBsimpsonsBsatisfyB	prejudiceBoffbrB	nostalgicBnicholasBmarsBlettersBlemmonB
lacklusterB	introduceBinferiorBgateBfoulBedgyBduckBdistinctBclicheBaustenBalcoholB1979BwouldbeBwintersBvoyageBviceB	upliftingBtripeBtomorrowBsleazeBsettleBschoolsBratBpsychicBplightBpassableBleoBkyleBjewsBjesseB	integrityBhumbleBhackB
complaintsBcitiesBchoosesBchainsawBbullockB1968BunawareBtribeBteachesBtaxiBsoundingB	secondaryBsagaBrupertB	respectedBrequireBreleasesBratesBraisesBoutfitBorderedBnonsensicalBmuchbrB	minutesbrB
landscapesBhungryBguitarBgregBgarboBexorcistB
disappearsBdespairBderangedB
definitionBchicksB
challengedBcentersB
apocalypseBaffairsB23BwheresBvisitsBvisibleBunfairB
underworldBtobrBthoughtprovokingBspeciesBslaveBsimultaneouslyB
simplicityBshowdownBrudeBrobbinsBreceivesBraymondB	preciselyBpeakBnowbrBmerylBfemalesBfeastBdoomBdefendBdeeBcreditedBcreamBcomfortBchildsBcharacterbrBboomBawfullyBantonioB13thByearsbrBviewingsBsixtiesB	senselessBsandBromeBresembleBquestionableBpatriciaB
passengersBpagesB	objectiveBnatalieB	musiciansBmundaneBmiracleB
middleagedBkubrickBharmlessBflairBexplorationB	excessiveBdebateBcubaBcrawfordBcorpsesBcakeBandreBaimBabsentB1983BworkerBwifesBtroopsBtoysBtastesBsinkBseverelyBrelyBrefusedB
popularityBphotoBoverlongB	operationBmoronicBmiseryBmasonBhopelessBdianaB	corporateB	conflictsB	companiesBchampionBbayBaussieB
activitiesBaccuracyBvoicedBunseenBunhappyB	testamentBsubparBstinksB	slaughterBrompBrexBrandomlyBpunB	pricelessBmethodsBlivelyBkickingBjordanBjackassBinvitedB	inabilityBgreedBfortyBfavourBexperimentalBevansB
equivalentBearBcrueltyB	crocodileBconcernBcomplainingBcatchingBcarlosBcancerBblastB	addictionBaccessBabusedBwidmarkBunpredictableBthompsonBsingersBsheenBseldomBscreensBretiredBpursuitB
presentingBplanesBpgBoriginBnerdBnailBjumpedBjoinsBhungBharmBfoxxBfifthBexperimentsBdysfunctionalBdiehardB	compelledBagendaB1990B
unsettlingBundeadBtriangleBstaticBshouldveBsarandonBrourkeBrepresentationBrecommendationBpurposesBpg13BparanoiaBodysseyBmuteBminusBmanipulativeBlandingBinterestinglyB	imitationBhelloBgemsBfirmBfemmeB	encourageBdivineBdislikedBdickensBdelicateBdefinedBconfessBcapitalBbonesBaltmanBabusiveBwagonB	streisandBrootsBregionBreachingBpreyBphillipsBobjectsBmollyBmessedBmasterpiecesBmailB
literatureBkathyB
insightfulBhokeyBhitmanBgreedyBgodzillaB
frightenedBfirstlyBdisneysBdesiresB	celebrityBcasualBbutlerB	biographyBbeverlyB1997BwidelyBvisitingBvelvetBteamsBtackyBsungBsneakBsamuelBsalmanB	resourcesB	repeatingBregardedBreelBpurpleB	prominentB	occasionsB
obligatoryBmetaphorBliftedB	holocaustBfeelgoodBdylanBdoubtsBcountsBclipBcaryBbiasedBaddressBvirtualBvinceBunfoldBsimmonsBrightbrBreevesBpreachyBperryBoverlookBnbcBmaidBliteraryB
expositionB
exceptionsBdressesB	dimensionBdevilsBconanB	computersB
complexityB
compassionBcluelessBbrooklynBberlinBanalysisBwweBwongBunexpectedlyB
transitionBtigerBspiritedBsnipesBshakespearesBsamuraiBridesBrenderedBrelaxBrecycledB	nicholsonBmuppetBmachinesBlilyBlasB	ignoranceBhughesBhippieBgriffithB	fantasiesBentitledBcloseupB
christiansBblobBbathBaweB	ambiguousB1971BwornBsymbolicB	strongestB	sillinessBshoulderBsecretlyBrouteBrossBroofBremoveBrealmBpalBnationsB	murderingBmodestB
misleadingBmentionsBliftBlengthyBlawsBkayBitemsBireneBinvestigatingB
inevitablyB	immediateBgrabsBfistBfillerBfeedB	exploringBexploresBearlBdomesticBdesignsB
describingBdameBcoBbronsonBassumingBvBsufficeBstuartBsnlBreferredBproB
positivelyBnodBmusicianBmilkBlosersBlinersBlesBkittyBinteractionsBhitchBharrisonBgoodbyeBgolfBfoolsBfalkB
directionsBcrispBcreatorB
confidenceBconceptsBchessBcemeteryBbetrayalBbeanB	barrymoreBbanalBatlantisBasylumBamusedB19thBwhollyBtrendBsuspendBsignificanceBshoddyBsharesBremadeBpromB
portrayalsBpolishedBorleansBministerBlocatedBlatinBjulianBjewelBjennyBinjuredBhindiB	hackneyedB	gentlemanBforcingBfirmlyBdrinksBdemiseBdaybrB	completedBbesideBbelaB1988BwrongbrB
unlikeableBthreadBthingbrBsurfingB	subjectedBstunnedBstumbledBstaleBsometimeBrisesBriotBrichardsBrespectableBremarksBremakesBrealisedBpaleBmutantBmillBloyaltyBkramerBinexplicablyBimprovementBhersBgabrielB	fashionedBerikaBdroveBdonnaBdifficultiesBdementedBdanesBdamonBcrossingBcoincidenceBchorusBcastsBcampaignBbudBbravoBbewareBballetBbabeBacidB1000BuweBunrealB
suspensionBsunshineBstatueBsobrBshoutingBshakyBservingBsandyBrestoredBreplaceB	reluctantBregardsBrefuseBqueensBpolishBpilotsBpennyBnoisesBmegBmayhemBlegacyBknockedBkarloffBiraqB	imaginaryBheroinB	evolutionBethanBdawsonBcubeBconradBbradyBattendBassignedBappropriatelyBapeB1993B1978BB	unlikableBswimBsunnyBseriesbrBsendingBresidentB	primitiveBpostedBownersBotherbrBnetBnemesisBmeritsBmadebrBlockBkeenBjoeyBjacketBinvitesBinsipidB
hopelesslyBensuesB
destroyingBdeletedBdaddyBcormanBconsiderablyBcommunicateBchuckleB
basketballBbargainBballsBanywaybrBanyonesBadoptedBactiveB
acceptanceB1987BwindowsBwigBwellwrittenBtravestyBtransformedB	threatensBsupplyBspockBsherlockB
respectiveBquietlyBoutingBorsonBolivierBoliviaBnewlyBkissingBkapoorB
imaginableBiconBherosBgrainyBfreaksBfiftyBeternalBelsesBdirtBdevicesBdestroysB	demandingBdanishBcouchBbrosnanBassumedBwizardB	witnessesBwhaleBwarriorsBwardrobeB	unrelatedBstringsBsoonerB	selectionBscriptbrBredeemB	profanityBpreposterousBplacebrBoutfitsBmontyB
mechanicalBmanbrBmacbethBlastingBlanceBinventedB	interestsBincidentallyBgroundbreakingBeugeneBethnicBdebbieBcrashesBcontinuallyBchillsBbusinessmanBbunnyBauthorsBaspiringB1984ByoungestBwooBtapBsurgeryBstudyingBsteadyB	speciallyBsomedayBsmilesBrodBrecordsBquentinBponyoB	policemanBphotosBpanBoptionBmonkBmarkedBinspireBinfectedBhearsBgloriaBgeorgesBfestBfascinationBemergesB	disgustedBdifferentlyBdesertedB
derivativeB
controlledB
conscienceBchoreographedBbuffaloBbabiesB
additionalB75B1982B
weaknessesBunexplainedB
underneathBsydneyBstillerB	septemberBscrewedBscreenwritersBronaldBrefersB	preferredBpoisonBpoemBpigBparadiseBminorityBmannBlocalsBlaurenBinconsistentBhomerBheightsBheavensBhatsBflashesBempathyBdrakeBdenzelBdemonstratesBdarnBcueBcopeB
convictionBchristieBchampionshipB
challengesBauthoritiesBarthouseBambitionBaidsBabruptB1998B1969BwretchedBwrapBwitchesBvulgarBunderstandsBtopnotchB
togetherbrBsubtlyBsharedBshakingBrubyBrooneyBreadersBranchBramboBracesBpuzzleB
psychologyBpotBpeteBownedBoffbeatBniroBmessyBmatchedBleesBinternalB	identicalB	heartfeltBgingerB	generatedBfrancesBfiBexposureBenteredBdistributionBconfrontationBchoppyBawaybrBattorneyBastaireBassassinationBangstB	acclaimedByardBwingsBwalshBvisionsBvainBtravoltaB	tastelessBsparkBsnuffBsnowmanBservantBpushesBpreviewsBpostersBpiratesB	partiallyBownsBorangeBnolanBmarvelBmarineBlimitBlenaBinsanityBhulkBhidesBdressingBdeceasedBdealerBcrittersBclaimingB	christinaBbostonB	amusementBalertBakinBagencyB1974B–BvibrantBveinBtopsB
terminatorBstareBsamanthaBritterBrippingBreaderBraisingBprovocativeBpromoteBpokerBpartnersB	paintingsBpacificBopportunitiesBnewmanBmuppetsBlethalBinsistsBhoBhistoricBheartedB
guaranteedBgereBfuriousBframedBdobrB
discussingB	dependingBcrazedBcravenBbugB	blatantlyBbamBaboardBwillieBwellmadeBventureBtunnelBtemptedBsurvivorBsurroundingsBsumsBsuburbanBsimpsonBshoreBsharingBrejectedBreferBquaidBpunchesBprequelBpitBmythBmichaelsBmasksBmapBlunchBleonBhustonBhopkinsBhilarityBhandlesBfiresBfiancéB	evidentlyB	energeticBdroppingB
difficultyBdevastatingBdanielsB
continuingBconnectionsBcommitsBbountyBbmoviesBattractBweatherBweaknessBwarmingBwakesBvastlyBtraitsBtailBsyndromeB	stretchedBstonesBstandupBshorterBsergeantBrussiansB
phenomenonBmatesBmachoBlayersBlastsBjadedBivBiqBhugelyBhostageBhammyBfluffBfilthBfillingBdurationB
detectivesBdemonicBcheekBbrendaB	afterwardB	admirableB1981B1970BvotedBvoightBvitalBuncutB	terrifiedB	supportedBstumblesB	strengthsBstoppingBskeletonBshootoutBropeBrolebrBriversBreflectsB	referringB	receivingBlabelBjealousyBintentionalB
indicationB	housewifeBhealthyBhayesBhangsBglowingBgamblingBgainedBfayBegyptianBeditBdreckBdeafBcontributionB
containingBcombinesBcloneBclichesB
caricatureBbombsB
astoundingBassureBaddictedB95B21stB200B1973BwesleyBvictoryBveraBtrampBtapedB	startlingBslipBsliceB	sincerelyBsciBrhettB	residentsBpleasingBpalaceBolBmotiveBmeltingBlaborBkidmanBjulietBhomesBgriefBgenderBgatherBfuryBfisherB	exquisiteBerrolB
engrossingBeliteBdunneBdoseBcoupledBconveysBcaricaturesBcannesBbilledBalecBactingbrB
accomplishB	absurdityBabruptlyB1994BtopicsBthievesBsitcomsBshowtimeBshortcomingsB	separatedBscoobyBschlockBrhythmBrerunsBrapidlyBplayboyBpiperBparadeB
motorcycleBmoronBlionelBlifelessB
kidnappingBjacksonsBinstitutionB
inaccurateBhootBhelpfulBhamBgirlfriendsBfunbrBfordsBfiringBfiftiesBeventualBearnBdismalBcostarsBconvenientlyB
confrontedBconfrontBchoosingBcarmenBbudgetsBbtwB	brutalityBanticipationB1945BvisitedB
undercoverBtoniBsiblingsB	shouldersBreignBratsBracingBprogramsBpredecessorBprayBnorthernBnieceBnicolasBmodestyBminsB	mentalityBmarriesB	justifiedBjudgesBinfoBhoganBhelplessBhaplessBhandlingBgregoryBfrozenBfeatB
farfetchedBeverbrBenthusiasticBenteringBdiazBdesignerBcusterBconventionsB
commentingBcolumbiaBclaustrophobicBchristianityBcareyBcanceledBbullyBbiopicBbarsBapproachingB
adolescentBaccountsB99B
wildernessBwaltB	victorianBtrialsBstudiesBsaraBroommateB	programmeBpainterB	nightclubBmuslimBmildredBmalesBleatherBlaurenceBkungfuBjudgmentBjillB	intricateBhainesBgreatbrBgoldbergB	extensiveBdidbrB	deservingBdependsBdaltonBcenteredBbonnieBbillsBbiblicalBavidB	attackingBakshayByepBwashedBtubeBtremendouslyBthoughbrBtechnicolorBsublimeBspencerBslimyBskitsBshtBshawBseymourBrewardBrescuedBrangersB
psychopathBnyBneuroticBnephewBmoralsB	misguidedBmeantimeBlouiseB
industrialBimpliedBgypsyBgrinchBgluedBgalBfogBfishingBexteriorBexaminationB	establishBendingbrBdudBdonebrBcrackingBcookingBcollectBchargedBcaliberBbogartBbikeB10brByoutubeBvotesBtrainsBsurvivesB
suggestionBstoresBstagesBshoppingBshadesBscrewBscotlandB	publicityBpierceBpaulieBoutrightBmitchumBmartyBitselfbrBingridBhavocBflashyB	explodingB
expeditionBexpectsBenhancedBemailBedieBdownbrBdisguiseBdisabledBdelBdeclineBczechBcushingBcornB	convincesBcomparisonsBbrosBbluntBactivityBwilderBwiderB	uniformlyB
translatedBskullBsidBsellersB	satiricalBrespectsBresolvedB	realitiesBrandolphBqBpaxtonB	orchestraBnunBnativesBmiyazakiBmindedBmercyBlongestBkansasBjarringBinteractBinhabitantsB	incidentsBimmatureBhornyBhkBhighwayBhestonBformedBfeministBeternityBdorisBdiscoBdillonB
despicableBdepthsBdefeatedBdeerB	confidentBcoloursBclooneyBcleaningBboastsB	backwardsBbacksBavoidingBasiaB	armstrongB4thBwiselyBwelldoneB	wellactedBwanderBwaitressBvoodooBvocalBunsuspectingBtossedBtorontoBtapesBsuitablyBstabbedBsopranosBsailorBraveB	principleBprestonBpornographyBoutlineBmormonBmorbidB
misfortuneBmealBloyBletdownBhydeBhomosexualityBhomicideBgrassBgalaxyB	fortunateB	fastpacedBescapingBdelightfullyB	convictedBconcludeBbutcherB	breakdownBacceptsB1991BwartimeBwagnerBusersBunimaginativeBtowersB	tolerableBtokenBthrilledBtacticsB	submarineBsteerBsessionBsensibleB	sarcasticBsafelyBrootingBrespectivelyB	regularlyBrecognizableBrealisticallyBprovingBpirateB
perceptionBpayoffBobtainBlimitationsBkarlBinstinctBheightBhankBframesBelmBdumberB	disguisedB	departureBcycleBcobraBcoatB	candidateB	camcorderBbreedBbeltB	befriendsBbaconBautomaticallyBauditionB
audiencebrBartworkBartsyBarrestBalteredBwastesBvehiclesBtonB
sympathizeBsymbolBstickingB	spectacleB	sentimentBsensebrB
richardsonB
restrainedBreminderB	relevanceBrainyB
professionBpredictB
possessionBoutdatedBnormBnailsBmorallyBlushBlivBlifesBladderBklineBjuryBidolBgrowthBfunctionB
foundationBexpenseBexpectationBeverettBelevenBdocumentB
disastrousB	decidedlyBdanaBcontroversyB
contributeBcomedybrBcentsBcattleBbeholdBbarrelBarkBamidstBaircraftB
accessibleB1989BwandersBvolumeBtriviaBtravisBtoolBtipBticketsBthrustBtcmBtaBswingBstudiedBstalkingBspiralB	spidermanBsoupBscoredBsaleBrollsB	rewardingBpredictablyBpenguinBpcBnaughtyBmixingBmallBlovebrBlegendsBknowbrBkidnapBjulesBinherentBgripBfortBfadeBexitBemployedBdubBdriveinBdenyBdeniroB	delightedBdallasBculturesBcrashingB
commitmentBcasinoB	brainlessBatomicBanytimeBalleyBafghanistanBaddictB21B1986B101BwendyBupsideBtowerB
stunninglyBsteamBstalloneBsinkingBseemingBseedB	screwballBscratchBsaintBrothB	repulsiveBpursuedB	publishedBpoppingB
phenomenalBpfeifferBpassageBnominationsBnetflixBminBmesmerizingBlowkeyBleadersBjanetBisraelB	ingeniousBimmortalBimmenseBgrabbedBgiganticBghouliesBfullerBforbrBerrorBelsebrBelevatorBdivorcedBdepictBcreepBcrapbrBcowBcountyB
colleaguesBcheerBcharacterizationsBcaseyBbulkBbroodingBbootsBbittenBbitingBauthenticityBaffectsB
advertisedBadoreBwtfBwardenBtoothB	suspectedBsupremeBstumbleBstairsBspreeBsketchBrussoBreviewedBreportsBrebelsBrearBposingBpoliticiansBpointbrB	paramountBobservationBnopeBnolteBneatlyBmontanaBmirrorsBloweBjulyBintroBinformativeBindianaBhippiesBhayworthBhabitBgodawfulBgerardBgatesBgapsBfillsBdubiousBdiverseBdisgraceBdevitoBdestinedBcelebrationBbuseyBbowlBbackbrBarcB
approachedBappealsBansweredBamazonBacknowledgeB300B1976B1950ByouthfulB	unnaturalBtendencyBtabooB	suspicionBsubsequentlyBstellaBspanBshiftBshanghaiBromeroBrifleB
relentlessBpsycheB	preparingBphysicsBowesB	overblownBoriginsBoldsBoldestB
noticeableBmedicineBmaximumBlynnBleapB
improbableBhepburnBheavyhandedBhauntBhallmarkBgratefulBgoodsBgilbertBghettoBfreakingBforgivenBfollowupBflowersBfarrellBfamilybrB	explosiveB	estrangedBemergeBdreadBdrabBdowneyBcriesBcounterBcopiedBconsiderationBconcentrateB	christineBchopBburstBbitchBbetrayedBbentBarrivingB19BwoundsBweakerBvileBughBtripleBtokyoBsubwayBstrokeBspeechesBsosoBshatnerBsfBseebrB	scenariosBreservedB
resemblingB	repressedBrealizationBradicalBpromotedBpretendsB
pedestrianBpeacefulBpatternBparanoidBoverbearingBothersbrB
moviegoersBjoannaBitemBinexplicableBindependenceB	incorrectB	incapableB	gentlemenBgearBgableBfrontierBfoolishB	endlesslyBeducatedBdeputyBdatesBcommunicationBcollinsBcbsBbwBboxerBbestbrBbelushiB	associateBashleyBantsB22B1977B1933BwireBwashBunsureBunsatisfyingB
threatenedBtheatresBsuckerBstylizedBstorysBspiderBsorelyBsmellBreliableBquestioningBpeckBominousB
noteworthyB
montgomeryB	monologueBmiloBmiikeBmiamiBmerchantBmensBmccoyBlongtimeBliteralBinmatesBgravesBglobeBglanceBfeedingBfataleBelviraBdumpedBdooBdolphB
conventionBconfuseBchaplinsBcentreBcarellBbuysBaxeBassuredB	architectBapplaudBamitabhBaimingBaceB	acceptingB1940B	valentineBsweptB
stepmotherBstardomBslavesBsharonBselectedBseatsBsaltBromeoBromancesB	remembersBrecoverB
rebelliousBqualifyB	practicalBpoundsBpossessBpoleBplottingB
paranormalBpadBmissileBmindbrBmidstB
mediocrityBmaBlongingBloisBlesterBlastlyBjohnsBincestB	immigrantB	horrifiedBgarnerBgangsBgadgetBflowerBfilthyBensureBembarrassinglyBdisgustBdestinationB
deliberateBcyborgB	criticizeBcrawlBcompeteBchoppedB	breakfastBbowBbenefitsBbeattyBattendedBapplyBalmightyBadaptionB	absorbingBabortionB85BwhiningBuniformBturdB	translateBtraceBtomatoesB	telephoneBsweepingBseniorBsellsBsandersB	rochesterBriceBrainbowBpsychedelicBpromptlyB
profoundlyB
principalsBpokemonBplantsBphotographsB	patrioticBpathosBnoticesBnewerBmosesBmillionaireBmeteorBlinkedBinterviewedB	intenselyB	insuranceBinformedB	inclusionBhybridBhiresBhiltonBhandheldBgoodlookingB	glamorousBgarlandBfulcisBfrogBflamesBfadedBexposeBexploitsBexperiencingBexcusesBexcruciatingBentriesBdominoBdividedBdisorderBdetractBdeterminationBdefiesBdazzlingBdarrenBdamagedB
convenientBcontestantsBcompositionBcloudsBclockBclimbBbridgetBboogieBblamedBbelleBbeforebrBbanterBbabesBaztecBaugustB
assignmentB1959B1939BzBwrightBwhinyBusbrBunsympatheticBunderstatementBtherebyBswallowBstuffedBslimBskilledBsixthBsissyBritualB	reviewingBrepliesBquarterBpursueBpierreBpenelopeBpeculiarB	parallelsBothelloB	offscreenBoceansBnarrowBmouthsBmenciaBmedievalB	macarthurBlynchsB	lightningBlighterBkinnearBjawBjanB	isolationBheelsBgorillaBgoalsBgesturesBflavorB	finishingBfarmerBeggBdaytimeBcreekBcostelloBcontributedBconsB
comprehendBcharmsBcabBbuzzBboringbrBbondsBbombingB	backstoryB
approachesBalvinBagonyB2008BwhoreBveteransB	underwearBultraBtitularB	throughbrBsugarBsterlingBstationsB	spoilerbrBsparksBsoughtBsomethingbrBshiftsBsanityBroundsBrobbersBrememberingBpraisedBparksBoverbrB	organizedBoffendBofbrBmurkyBmickBmasterfullyBmarlonBmarinesBlendB
legitimateBjulietteBisabelleBinsightsBinjuryBfrightBfreakyBexcessBevokesBenBdramaticallyBdragonsBdownfallBdictatorBdecidingBcurlyBcritiqueBcoveringBcoreyBconveyedBcontrolsBcliveBclassyBclanBcinemabrB	cigaretteBcheaplyBchaoticBcastbrB
cartoonishBcaperBcampusBbeatlesBbalancedBarabBappliedBallensB
aggressiveB	aestheticBadditionallyBachievesB1985BvomitBvividlyB
villainousBverdictB
unfamiliarBswitchedBsubtextBstrictBstargateBstableB	spaceshipBslyBseveredBscariestBsaneBrugbyB	redundantB	possessesBposesBpointingBpamelaBotooleBorganizationBnutBnervesBnerveBlionsBlikeableB	inventionBignoringBhookerBhitlersBhintedBhansBgroundsBgandhiBflagBexpertsBencounteredBembraceBelectionBdustinBdumbestBdinerBdilemmaBdiamondsBdeathbrBdadsBcorporationB
collectiveBclichedBclashB
chroniclesBcharacteristicsBcasperBcarnivalB	cameramanBboyleBbikerBarguingBalaB	aftermathB	affectingB1936ByoursB
youngstersBwaynesBwarrantBwarnsBvoidB	unusuallyBuniformsB	tormentedBthreadsBswingingBswearingB	stephanieBstartersBsourcesBsmashBsleepsBslappedBsensationalBselfindulgentBseasonedBroughlyBresultedB	remindingBproductsBpotterBpolanskiBploddingBpalanceBottoB	originalsBoldfashionedBmutualB
mannerismsBmacabreBlupinoBlolaBloBliBlenoBjudgedBinsultsBinspirationalBindifferentBiconicBgibsonBfundingBforemostBfoilBflowsB	firstrateBexplodeB
executivesBemployeeBelliottBegyptBedwardsBeagerlyBduvallBdumpB	deliciousBdebraBcrookedBcontemptBcharliesBcapacityBbeggingBbattlingB	assembledBadmiredBacquiredBwhitesBvalB
undeniablyBunconventionalBuhBtvsBtireBswayzeBstereotypedBsportingBsleepyBskitBskinnyBsketchesB	sickeningB
shockinglyBsentimentalityB	seductiveBsangBrightlyB
resistanceB	reflectedBreBrajBragingBportmanBporterBpizzaBphillipBpansBnathanBnarratedBmumBmessingBmelodyBlocateBkoreaBjoBjerseyBisraeliBintactBinsistBinsertedBhystericallyB	himselfbrBheatherBharborBglaringBgarageBfloodBfactualBfactorsBeyreBexploitBestablishingBentiretyBebertBdistractionB	discussedBdemonstrateBdefineBcobbBbreadBbillingBbakshiBawfulbrBartyBanyhowBabrahamBaboundBwayansBvaughnBvariedBvanityBvanessaBumB
suggestingBsubstantialBslideBsleeperBshocksBseedyBramblingBprosBnortonBmusicbrBmoronsBmitchBmamaB	maintainsBmadsenBlonesomeBlavishBlaterbrBkeysBgloomyBglenBfuelBfragileB
forgettingBfontaineBexclusivelyBevanBduelBdistinctionBdisappointedbrBdetroitBdebtBcohenBclubsBclerkBclausBchuckyB	breathingBbettieBbellyBbegBbarkerBbaddiesBassumesB	argumentsB	ambiguityBadsBactorsbrB34ByawnBwheelBvivianBverbalBupdatedBtongueincheekBshuttleBshelleyB	shamelessB	sensationBrustyBroundedBreverseBrenaissanceB
protectionBoharaBobserveBnoveltyBnovakBmoviemakingB
meanderingBmarcBluisBlucioBlicenseBlansburyBjudeBjobbrBjessBhostileBhartleyBfritzBfodderBentertainingbrBearnestB	dominatedBdiveBdegreesBcrippledBcookieBconsistBcomplicationsB	colleagueBclayB
celebratedBboothBalterBafricanamericanB1920sBwhereverBvintageBupbeatBunderstandablyBtonesBthirstBtargetsBsweatBsurroundB	stretchesB
statementsBstagingBspineBshoeB	sentencesBseduceBsatisfactionBromanianBrivalsBrelaxedBreaganBrationalBpsychologistBprotB
politicianBpauseBpamBomenBnormaBnoahBnewcomerBnetworksBmonroeBmoeBmisunderstoodBmanipulationB	malkovichBlendsBleanBkumarBkellysBjoiningBitdBistanbulBinteriorB	graveyardBgimmickB	gatheringBgapBfluidBflimsyBfiennesBfiancéeBexcellentlyBeconomicBduBdoveBdiaryBcrossedBcowardB
confessionB
complainedBcomebackBcirclesBchillBcellsBborrowBboobsBbasingerBbaronBbanksB	antonioniBamirBadrianB1975B
witnessingB
winchesterB
werewolvesB
watchingbrB	viewpointBvergeBunBtransferredBtightlyBtaraBswedenBsummedBstonedBspadeBsmugBshannonBsatanicBrousingBrespondBrejectsBreidBrehashBredneckBrapistBprostitutionBpostwarBpollyBpocketBphaseB	perceivedBparticipateBpaintsBoctoberBmonkeesBmoneybrBmobileBmixesBlucilleBlooneyB	languagesBlandedBkermitBinviteBintroducingBinterestingbrBhollandB	harrowingBglimpsesBfruitBexperiencebrBexistingBexcruciatinglyBenduringBedisonBdriversBdodgyBdismissBdiBdespiseBderBdemonstratedBdeliciouslyB	considersBclimbingBclaraBcheersBcerebralBceilingBcarsonBcampingBbuttonsBbratBbordersBbooneBanxiousBantiheroB
antagonistB	addressedByearoldB	wholesomeBustinovB
unansweredBturmoilB	toleranceBtheoriesBspiesBspellingBsociallyBsinksBsettledB	sebastianBsarcasmB	salvationB	retellingBrelentlesslyBprogressionBplanetsBperformancebrBpeersBpbsBpalmB	overboardBopenlyB	obsessiveBmudBmotionsBmelvilleBmarxB	magicallyBmadisonBkrisBknocksBkathrynBjuanBinaccuraciesBidealsBhenchmenBguineaBgraysonBgrandmaBgiantsBgeekBfuzzyB	exploitedBenhanceB	emphasizeBdjBdixonBdiggingBdashingBcousinsB	consciousB
complimentB
committingBcocaineBcloudBcapeBbustBbennyBbennettBbendBbelievabilityBarrayBabstractB28B20sBwrestlerBwhybrBweaverBwakingBvcrBunattractiveBtaxBstatingBsleptB	sincerityBshelterBsensibilityBsegalBsalesmanBroutinesBroryB	robertsonBrampageBprofessionalsBproceedBpompousBplateBpattyBpathsBpacksB
overweightBordealBobservationsB	mythologyBmysteriouslyBmustveBmovieiBmotelBmomentumB
melancholyBlipBkeanuBkBiranBinvestigatorB	instancesBincreaseB
immigrantsBherculesBheadacheBgromitBfranticBfetchedBfeminineBepicsB	enigmaticBehB
distractedB
definitiveBdaylightBdaffyBcrosbyBconsequentlyBconsciousnessBcomaBcladBbiasB	attendingBassociationBabbottB1960B	variationBvalidBunconsciousBumaBtraumaBtombBtolerateBsustainB
substituteBsubgenreBsubduedBstreamBstewartsBstalkedBspitBshotgunBsergioBseBsabrinaBricciBresumeBreadilyB
progressedBpoundBpoeBparticipantsBpaltrowB
originalbrBollieB	officialsBoffenseBmooresBmirandaBmelissaBmandyB
liveactionBknockingBindicateBholmBhawkeBgoatBginaBghastlyBgenerateBgameraBflickbrBfeebleBexplodesBexpertlyBenoughbrBeggsBeconomyBeaterBeagleBdudleyBdistressBdistractBconsumedB	combiningBchristyBcharacterisationBcanyonBbsBbryanBbloomBaustensBarebrBalliesBadmittedBBzaneBxfilesB
wonderlandB	warehouseBunderdogBunderdevelopedBtrivialBtransportedBtodaybrBtimonBthugBthingsbrBthebrBswitchesBstuffbrBstreakBstabBspringerBsourBslashersBsalesBrodneyBripsBrewardedB	revolvingBresolveB	releasingBpolicyBphoenixBperverseBpatriotBoverusedBnuancesBneroB	neglectedBmyrnaBmermaidBjurassicBinterruptedBinfluentialB
incestuousB	impendingBheapBhawksBhandicappedBgrislyBgeraldBfranklinBforgivenessBflowingBflipBfacilityBexwifeBerBentranceBengineerBelectedB	eastwoodsBdrillBdownsB	disregardBdealersBdeadpanBcursedBcrownB
criticizedBcoppolaBconsequenceB
compensateBcoburnBchaneyBcarolineB
captivatedBblessBbishopBbikiniBbeardBaudreyBaffleckB73B1944BzanyBworsebrBwormsB	vignettesB
traditionsBthunderBtestingBspiceBsmarterBsignificantlyBscottsBsammyBromaniaBrobbedBrickyBrecklessBraidBpythonBprogrammingBproducesB	populatedBpoppedBpenaltyBpapersBpaBnivenBneedingB
nauseatingB	murderersBmichealBmaskedBlistenedB	libertiesBliarBlaunchBkentB	instinctsBinsertB
infinitelyBinclinedBinadvertentlyB
highschoolBgeorgiaBgalleryB	furnitureBforgetsBflippingBfenceBengineBelderB
dedicationBdaysbrB	customersBcunningBcourtesyBcontrollingB	collectorBcheeringBcheadleBcassidyBcampsBbuffyBbothersBboostBboardingBbittersweetBbegsBbartBatlanticBasterixBaptBandreasBampleBadmitsB1992BwinnersBweeklyBvaderBupcomingB	unfoldingBuncannyB
timberlakeB	surrenderBsuperfluousBstrungBstakeBsolvedBshepherdBshawnBsemiBscroogeBscreenedBrudyBritchieBrioBregimeB	recurringBramonesBpreciseBpeggyBpairingBoutsetB
optimisticBnutshellBniftyB
monologuesBmarcusBmannersB
lieutenantBknightsB	knightleyBinvestedBinformBillusionBhuntedBhowlingB	historybrBherbertBheathBgravityBgraduateBgoodmanBfixedB
enchantingB	employeesBedmundBdustyBdrumBdoyleBdistinctiveB
displayingB
depictionsBdamBcoffinBclickBbogusBbitchyBbearingBbattlefieldBbatsB	awarenessBarticleBanticipatedB	alexandraB	admissionB
admirationBadaptByellsBwovenBwhoopiBunclearBtierneyBteddyBsunsetBsundanceBsuicidalBsuckingB	subtitledB	stupidestBstupidbrB	spotlightB
similarityBsethBservicesBsensitivityB	satelliteBrogueBrecreateBrebeccaB	qualifiesB	pleasuresBpivotalBpinBphoebeB	pervertedBpainsB	obstaclesBnuancedB	norwegianBninetyBnielsenBnerdyBmythicalBmonicaBmockBmaloneBmalcolmBlurkingBlunaticBloganBkurtzBkristoffersonB	judgementBivanBislandsBinconsistenciesBhypnoticBhypedBhorizonBhoneyBhesitateBhatefulBgutBgrierBglendaBgeinBfarleyBfanaticBfacebrB	enchantedB
elementaryBebayBdukesBdodgeBdetachedB	depardieuBdeannaB	courtroomBcountingBconvictB
conflictedBcokeBcliffhangerBcircaB	centuriesBbuildupBbookbrBbasilBbacallBautoBartisticallyBarrowBaliciaBaidedBabandonB18thB	worldwideBwarbrBvotingBuneasyB	transportBtherapyBtheftBtautBstrippedBstackBsmithsBshadyBscriptwriterB	schneiderBrunawayBrobocopBriderBretrieveBrespondsBreplacementBrainesBpriestsB	premingerB	preachingBposeBovershadowedB
ostensiblyBoccupiedBninaBmoriartyBmobstersBmanagingBlumetBlibertyBlangeBlabeledBkubricksBkristinBkilmerBkeatonsBjuiceBirwinBhomebrBgreatsBgoldieBfreezeBfleeB	festivalsBfarmersBfairnessBfadesB
excellenceBdrawingsBdraggingBdevotionBcustodyBconclusionsBcoloredB	classicbrBcheerfulBchatBchaptersBcasuallyB
casablancaBbubbleBbikoBbachB	awakeningBantonBallyB	allegedlyBactionbrByellBwangBwaltersBupdateBthoBtablesBswampBsternBstalkerBstadiumBssBsophieBsidewalkBshylockBshockerBsheetsBsgtBsensualB	scriptingB
scratchingBrookieBrollerB
retirementBreneeB	renderingBrelatingBprologueB
principlesBpoirotBpoetBpimpB	picturebrBperiodsBparkingBpalsBmysticalBmortalBmomsBmillsBmidwayBmadmanBlizBlimpBlightweightBkurosawaBjudithBjoseBiraBinsultedBimitateB
identifiedBhooperBhometownBhappenbrBgillianBgenieBfundamentalBfrombrB	followersBfleshedB
flamboyantBfilmographyBfierceBfaintBestherBestablishmentB
engagementBdowntownBdistributedBdenisBdcBcynicismBcrossesBcreepsB	confrontsBconfinedB
comparableBcollapseBclunkyBclarkeB	civiliansBchowBburtonsBbrookeBbrettBbrendanBbreastBblinkBbleedBbeowulfBawaitingBauteurBartistryBarielB	ambitionsBalisonBachievementsB1967Bww2B	verhoevenB
unfinishedBtunedB	traumaticBtoxicBtechBsystemsBswansonB
surrealismBsurfBstrainBspringsBsniperBsnappyBsinsBsicknessBshempBsheilaBservantsB	semblanceBsectionsBruiningBrosesBreunitedB	restraintBrepublicBrenoBredgraveBrazorBpuertoBprotestBpickfordBphiladelphiaBperkinsB
officiallyBoccultBnoelBnightmarishBnestB
marvellousBmarketedBmarathonBlorreBloanBlistsBlelandBlambertBkiteBkindlyB	katherineBjacquesB	irritatedBinterpretationsBinhabitB
improvisedBideabrBhurryBhopB
hellraiserB
graduationBgatheredBfunkyBflockBfileB
favouritesB	fairbanksBernieBdynamicsBdistinguishedBdangerouslyBdandyB	childlikeBcheatBcharacteristicBcampfireBbrickBbreatheB
boundariesBblockbustersBblackandwhiteBbegunB
assistanceB	arroganceBaroundbrB	anonymousBadvancesBabroadB500B1957B1951B1948BzombiBwhoveB
wheelchairBwearyBvirtueB	underusedBthirtiesBsymbolsB
sufficientBsteeleBstarsbrBstanceBsmoothlyBshakesBseinfeldBseattleBscheduleBscarierBsaloonB
sacrificesBreluctantlyBrathboneBrangingBraidersBpropB
photographBperspectivesBpatchBparodiesBoverwroughtBorphanBoperateBomarB	obliviousBnerdsB	necessityBmobsterBmissionsBltBlorettaBlongbrBkidnapsBkathleenBjacksBignoresB
idealisticBhornBhardestBfulfillBfederalBdvdbrBdreamingBdishBdinB
denouementBcrushedBcrowdedBcowboysBclarityBchipBcassieBcaptiveB	cancelledBborrowsBbondageB	bodyguardBboardsBbachelorBarquetteB
armageddonBanchorBallegedB1934B
witchcraftBwarfareB	villagersB	vigilanteBvertigoBunstableBtroopersBtriggerBtransitionsB	terrorismBtellyBtalkyB
talentlessBsurgeonBsuaveBspinalBspawnedBsoxBsonnyBsightsBsighBsenBsecureBsealBschemingBrevivalBreportedBreeveBredfordBraysBrantBquintessentialBpuzzledBpubBprofitBposeyBpornographicBplatoonBplaguedBpigsBpartbrBpairedBownbrBoutdoorB	obscurityB
mulhollandB
monotonousBmeredithBmarryingBmanipulatedB	macdonaldBlordsBlizardBlengthsBlaysBlargestBkinskiBkatieBjolieBit´sBinterspersedBintendBinsomniaB	injusticeBhooksB	homicidalBhenchmanBhawnBguardianBgraffitiBgenrebrBfrontalBframingBeyedBexcelsBevolvedBdynamiteBdominateBdomainBdivisionB
distinctlyBdirecttovideoBdecencyBcubanBcrowdsBcracksB
commandingBcmonBclinicB	chocolateBcalBbutchBbucketBbombedBblazingBbearableBbcBbashBballoonBbaitBatrocityBantiB	announcedB	animatorsBamateursBalotB1953B1932B	zellwegerBworryingBwolvesBuptightB
transformsBtheirsBstoicBsteamingBspottedBshelvesBsheepBshamefulBschemesB	scatteredBrepresentativeBrelatesBrejectBrecipeBreasonbrBrealisesBrapesBramseyBramBprostitutesBprecodeBportionsBphonesBpaulaBpaddingB	overtonesB
outlandishBnotingBninjasB	moderndayBmagicianBlindsayBlensBlayingBlaurieBlangBlambsB
laboratoryBkinkyBkindnessBjaredBinformsB
incidentalB
impeccableBimoBimhoBgovernmentsBgoodingB	functionsBflewBfirthBfetishBfastforwardBfashionsBfailuresBevelynBeuropaBelectricityB	dreamlikeBdistrictBdesolateBdenyingB	deceptionBdavidsBcutterB	curiouslyB
courageousB	conveyingB
consistingB	confirmedBcindyBchevyBcharityBchargesBceremonyBcapB	boxofficeBbimboBbethBautobiographyBapplauseBangelinaBaerialBaccompanyingB1966B1955B150BwronglyBweeBwatcherBwarpedBvanillaB	unwillingBunnecessarilyBtruthsBtripsBtreatingB
transplantBtonedBsylviaB	sylvesterBsubconsciousBstirringBspoofsBspoilingBspectrumBslipsBskippingBshredBsexistBscorseseBscorpionBscoopBrukhBrisksBrickmanBrepeatsBremarkBraunchyBrandallBproportionsBproneBpreacherBpolarBparsonsBoutlawBobservedBnavalB	motivatedBmoldBmistyBmiddleclassBmcqueenBmarilynBluxuryBlovinglyBlopezB	lookalikeBlilB	lifeforceBlestatBkneesBkingsleyBjokerBjacobBintruderBinsanelyBimplyBhmmBhardenedBhalfhourBhairedBhahaBgundamBgovindaBgoshBgigBgeoffreyBgeeB
fulllengthBfriedBfratBfortiesBeyesbrBevokeBeffortlesslyBedgesBderivedBdemeanorBdeedBdeborahBdangersB	coworkersBclientsBchefBcheatsBcelebritiesBcavalryBcamillaB	butterflyBbrownsB	brazilianB
borderlineBbonBbleedingBblaxploitationBbasketB	backwoodsBavoidsB	assassinsBarrangedBappliesBadventurousBabominationBwebbBveronikaBvegaBvanceBunappealingBtrickedBtorchBtenantB
temptationBtackleBsunkBstripperBstrainedBsnowyBslowerB	showcasesBshovedBsharksBshapedBsentinelB	rosemarysBrevoltB
retrospectBrequestBrealbrBrailroadBpursuingBpressedBpopeB
playwrightBplayfulBpickupBoveractsBoutcastB
occupationBnearestBnailedBmisterBmiceBmerryBmanicBlorenzoBlivesbrB	lingeringBjigsawBjanesBhoustonBhostsBhopefulBhisherBherzogBhannahBgoofsBgestureBfraudB	fishburneBfalconBexploitativeBexplanationsB	europeansB	espionageB	emergencyBelliotBechoesBdwarfBdrownedB	distortedBdirectorwriterBdevelopmentsB	determineBdeskBdesBdelveBdarklyBdarioBdafoeBcrashedBcountedBcodyBclaytonBchamberB	celebrateBcarnageBcarlyleBburntBbrassB
boyfriendsB	bloodshedBblessedBberkeleyB
beforehandBbashingBbaldBassetBannoyBamidB
adequatelyB	abundanceB27B1943BwrestlemaniaBvibeBvaryingBvapidBtalkieB
supportiveB	skepticalBshoutsBserialsB	sentencedBsenatorBseagalsBruledBruggedBrichlyBregisterBrapidBradarBpsychologicallyBprankBpianistB	permanentBpercentB	pattersonB	passengerBparrotBpalmaBongoingBoneillBnigelBmockeryB	magazinesBloonyBlonBlockeBlinearBlesbiansBlaunchedBjustificationBjoyceBjerksBitaliansB
increasingBinchBimpliesB
identitiesBhermanBharmonyB
happeningsBgrinBgobrBgamebrBfriendshipsBfinneyBfayeBfartBfantasticallyBfamedBextraordinarilyBepisodicB
entertainsBelmerB
eisensteinBearliestBdrainBdependBdeniedBdeedsBdarlingBcummingsB
criticismsBconfinesB	condemnedBconcreteBconcentrationBcompassionateBcommendableBcombsBcohesiveB
classifiedBclarenceBchoirBchloeBcaligulaBbitesBbigfootB	bartenderBawardedBatticBattenboroughB
astronautsBarizonaBappleB	anthologyBamberB51B1941BwalmartBwaitsBveronicaB	unleashedBuncoverB	uncertainBtrustedBthurmanBtargetedBtaglineBswordsBsuppliesB	strangestBsteamyBstabsBspinningBsoberBsmallestBslaughteredBshearerBshamelesslyBseriousnessBrumbleBrewriteBreviveBretroB	remainderBrelyingBpunsB
protectiveB
protectingB
portugueseB	paragraphB	operatingBoddballBnovelistB	nothingbrB	miyazakisBlikebrBlifelongBliamB	lettermanBlandmarkBkruegerBkissesBjollyBjewBjerkyB	jeffersonBjamB	interplayBintelligentlyBinducingBiconsB	hungarianBhorrorbrBhilariousbrBherdBhairyBgielgudBexorcismB	episodebrBemmyBduchovnyBdriftBdistinguishB	creationsBcrassBcoyoteBcoupBconnecticutBcommentatorsBcometBcolemanBchucklesBchopsBchewBcaptBcandleB
braveheartBboutBbondingBbloodthirstyBbillieBbertBbehavesBbarberBbackedB	awfulnessB	astronautBargentosBappalledB
anythingbrBantiwarBanguishB	accompanyBacclaimB
accidentalBacademicBabsorbedB1963B010B
yourselvesByokaiByearningBwipedB	whimsicalBwhereinBvisceralB	unnervingB
undertakerBtrumanBtrapsB	torturingBtitsBswiftB
sweetheartBstraighttovideoBstefanBstapleB
slowmotionBskippedBshoBselfcenteredBscreenplaysBscarredBsassyB	sasquatchB
sacrificedBroadsBrescuesBreneBrelievedBrecallsBramonBpunishedBpsychiatricBprospectBprayingB	policemenBpistolBpennedBpenBpaddedBoverwhelmedBoutrageouslyBorientedB	opponentsBobsceneBobrienBnightbrBnatashaBmorseBmonksBmiraculouslyBmeaningsBmacBluridBlogBlawyersBlavaBlaserBkolchakBjokingBjointBjockBirresponsibleB
instructorBinnuendoBhunkBhockeyBhilaryBhackmanBgracesBgoldblumBgaspBforrestBfightersBfascistB
expressingB
encouragesBeleanorBearnsBdreamyBdogmaBdoesbrBdisappearanceBdimBdernBdeliveranceBdeclaredBdeckBdecemberBcrookBcoworkerBcounterpartBcostnerB	connivingBcoincidencesBchoreBchipsBcarriageBcapsuleBbrandonBbossesBbelieverB
battlestarBbastardBbarbraBavenueB	automaticBastonishinglyBanatomyBadoredB400B250B1958B1942BzoeyBwynorskiBworshipBvengefulBunravelBuninspiringBunforgivableBtraveledBtormentBtoolsBthroneBtendedB	switchingB
suggestiveBsteppedBstarvingBspearsBsorrowBsmackBslutB
skillfullyBshoutB	shootoutsBsg1BserumBschizophrenicBscheiderBscarletBrussBrougeBroboticBrobbingBrevelationsB	replacingB	reasoningBrampantBquartersBpuppyBprofileBplugBplatformB	placementBpenisBpaycheckB	overnightB	overactedBoutputBnannyBmishmashB
mercifullyBmangaBmackBlightlyBledgerBlaraBjediBjarB
investmentBinstrumentsB	insteadbrB	inhabitedB
influencesBinexperiencedB	indulgentBincompetenceBhuttonBhurtingBhmmmBhawkBhabitsBgrooveBgreeneBgrangerBgiggleBflashingBfearedBfarscapeBfarewellB	fairytaleB
enormouslyB
encouragedB	diversityBdisconnectedBdisappearingBdenverBdashBdaisyBdahmerBcrooksBcrocBcounterpartsBcontributesB
continuousBconfederateBconcordeB	competingB	communismBcollaborationBchewingBcarrotB	caribbeanBcareerbrBbumBbranchBbotchedB
biologicalBavengeB
associatesBarnieB	apologizeBanniversaryBallisonBallanBadmirerB1931BzhangByarnB	wrenchingBwipeB	weirdnessBwarnersBvolumesBvisitorB	violentlyB	versatileBupstairsB	ultimatumBthelmaBtestedBtempestBtemperBsupportsBstaresBstabbingBspectacularlyB
soderberghBslugsBsitesB	signatureBshinyBshahBshaggyBseussBsearchesBsaybrBsaintsBripoffsBreuniteBretainsBresurrectionBrepresentingB	reportersBrenownedBremorseB	radiationB	promotingBpremisesBprehistoricBpredecessorsB
powerfullyBpotentBpokeBpeaksBpartialBowningBoverseasBoperasBnunsBnapoleonBmormonsBmondayBmindsetB	materialsBlundgrenBlimbsBklausBkaufmanBjuddBitiB	interiorsBinjectB
illustrateBheartilyBhatingBharrietBhandyBhanBgymBguinnessBgoersBghostlyBgadgetsBfugitiveB
forebodingBfelliniBfannyBexamineB
everybodysBentertainerB
electronicBedithBdurbinBduhBdruggedBdrifterBdosesB
directorbrBdexterBdestructiveBdeneuveBdarthBcuesBcropBcorrectnessBcorbettBconsistencyB	complainsBclosureBclientBcheerleaderBcedricBcecilBcannibalismBburkeBboyerBbanditBawryBauraBathleticBarchitectureBanxietyBantwoneB
animationsBamoralBambianceBadulteryB	admirablyB	achievingBaccomplishmentB55B26B1938B
yourselfbrBwwiBworriesBwillardBviscontiBvinnieBversaBusageB
uncreditedBtruthfulBtrendyBtransparentBtilB
thereafterBtheodoreBswissBstylebrB
structuredBstillsBstarshipBstakesBspontaneousBspacesBsolvingBsirkB
shoestringBsailorsBrosarioBrhymeBrevolveBrefugeBreformB	rebellionBpunksBpreventsBpresumeBpremB
preferablyBpranksB
possiblebrBplotlineBpattonBoweB	occurringBnuanceBnovemberB	northwestBnaschyBmutedBmuslimsBmuscularB	momentsbrB
moderatelyBminesBminersBmentorBmashBmarquisB
manipulateBmabelBlawnBkomodoBkleinBjunkieBironsBiranianBinvestBintendsB
imprisonedBilBidiocyB	hypocrisyBhunkyBhumB	honeymoonBhiphopBhiBheroinesBhealingBhassanBgripsBgretaBgreeceBgrandeurBglossyBgeniusesBfreddysBfleetingBflameBfiendBexpandedBexistentialBenvyBenthrallingBensueBenglundBearthsB
earthquakeBdudesBdolemiteB	documentsB
disciplineB	disastersB	dependentBdenialBdemilleBdeepestBdeemedBcynthiaB
cronenbergBcrackedBcosBcoolestBconvictsBcontinuouslyBconstraintsB	consistedBconnieBclawB
censorshipBcautionBcastedBcargoBcagesBbyrneB	butcheredBburstsBblamesBbgradeBbergmansBbathtubBateBashBapproximatelyB	annoyanceBallegoryBagobrBwrathBwillsBvolcanoBvirtuesBvillaB	videotapeBunsuccessfulBunremarkableBunluckyB	transformB
tragicallyBtobyBtillyBthatllB
terriblebrBtepidBtensionsBteamedBswitzerlandBsuzanneB	surpassedB	superstarB	stumblingB	struggledBstereotypingBsteppingB
standpointBstalksBsophisticationBsmarmyBslowsBshebaB	shatteredBshakespeareanBschwarzeneggerBsaddestBsackBrollercoasterBrolesbrBriffBreelsB	ravishingBrabidB
pronouncedB	priscillaBprintsB	prevalentB	predictedBpassiveBounceB	negativesBnarcissisticBmythsBmovieandB	monstrousBmockingBminnelliBmillandBmichelBmessbrBmclaglenBmcdowellBmccarthyB
maintainedBlectureBleapsBkahnBinterpretedBinsignificantBillustratesBhostelBgrosslyBgrandparentsB	geraldineBgeneticB	friendsbrBfreakedB	forgivingBfleesBfleeingBfinishesBfilesBfableB	expressesBenduredBemotionlessBelusiveBelijahBechoBducklingBdrippingBdramabrBdirkB
deservedlyBdefyB	defendingBcurtainBcryptBcowroteBconcentratesBcockyBchemicalBcensorsBcaronBcainBbranaghsBblokeBbehalfBbaxterBamokBallstarBalliedB	alexandreBalainBadeleB007ByaddaBwoefullyBwidowerBwahlbergBvulnerabilityBvivaB	versionbrBvaultB	unnoticedBturtleB
travellingBtouristsBtossBtinaBthunderbirdsBthumbBthereofBthereinBtastyBtackedBsyncB	surpassesB	summarizeB
successionB	stylisticBstinkB
stalingradB
spielbergsBsnapBslippedBsearchedBscarfaceB
scarecrowsBsampleBrootedB	rooseveltB	rodriguezBroachB	revoltingBrefusingB
reflectingB	recoveredBquantumB	privilegeBpricesB
prejudicesBprefersB
postmodernBposseB	positivesBpodBpenchantBpartyingBpalpableBovertlyB
orchestralBoptimismBoprahBoctopusBnoticingBneutralB
neighboursB
millenniumBmidgetBmartiansBmanufacturedB
mannequinsBmaeBlyricalBlukasBlowbrowBloopBlocalesBliuBliftsBknackBkeeperBjodieBitvBinterrogationBintellectuallyBimplicationsBidaBhustlerBholdenBhiringBharvestBguidanceBgrandpaB	goodnightBgoddessBgilmoreBgarneredB
fulfillingB	fulfilledBformerlyB
fictitiousBfamiliarityBemployBdutiesBdreyfussBdopeyBdnaB	dismissedBdietrichBdienB	conductorBconcentratedBcomparesBcoasterB
clevernessB	cleopatraB
classmatesBcivilianB
chupacabraBcheungB
categoriesBbynesBbumpBbritBbrentBbreakthroughBblendsBbillionBbenjaminB	beethovenBbavaB	barbarianBbaffledBbadnessBbaddieBbadassBatwillB
atrocitiesBassistBarmorBanilB	andersonsBamritaBamiableBallianceB	alejandroBalarmBadaBactorsactressesBactionpackedB1946B1930BzealandByBwwfBwokeBwildlifeBwendigoB
voiceoversBvargasButahBundevelopedB
undeniableBtwodimensionalB
transcendsBtonysBthenbrBthemedB
tendernessBteaseB	surroundsB
subversiveBstimulatingBstardustBsqueezeBsparedBsordidBslobBsiegeBshiftingBsensibilitiesB	seductionBsealsBscreamedBscandalBrivalryBripleyB
rightfullyBriddenBrewindB
recreationB	realitybrBraoulBrantingBpryorB	prolongedBprolificBprofessionallyB	premieredBpostapocalypticB	positionsBplacingBpioneerBpicturesqueBperilBperformancesbrBowlBoutbreakBoptionsBnevskyBmeadowsBmathieuBmarvinBmarvelouslyBmanosB
managementBmaintainingBlureBlonerBlombardBlauBkazanBjessieB
invincibleB	interpretBinsistedB	inheritedB	indicatesBhuppertBhumiliatingBhugoBhowardsBhavebrBhaggardBgossipBgoslingBgiB	galacticaBfreddieBfleetB	firsttimeBfauxBfathomBfabricBexternalBexistentB	exhaustedBerabrBentertainmentbrBemilioB
eliminatedB	efficientBdwightBdreamedB
dreadfullyBdrainedBdownbeatBdopeBdominicBdistastefulB	dimwittedBdataBdaftBcursingB
cunninghamBcsiBcravensBconnorBconfrontingBconductB	clockworkBcircuitBcherryBcheaperBcharltonBcatwomanBcathyBcasebrBcarpetBburyBbsgBbrushBbigotryBbelievablebrBbarnBbardemB	awkwardlyBauntsB	attendantB
assortmentBariseB	arbitraryBapocalypticB
alienationB	addressesBabrB510BzizekBwidowedBvesselBvalerieB
upbringingBuniquelyBuncompromisingB	tragediesBtollB	throwawayBtextureBtestsBtennisBtediumBtakashiBsweeneyB	suspendedB	superiorsBstirB
sophomoricBsondraBsnatchBslumberBslangBslamB	silvermanBselleckBscumBscoringBsciencefictionBrowlandsBrestlessBreliedBrefugeeB	recogniseBratioBrangesBrackBquaintBpuzzlingBproductionbrBproblematicBpepperBpausesBpatheticallyB
newspapersBmustacheBmotifBmeyersBmementoBmeandersB	mastersonB
mastermindB
marginallyBmanneredB	mandatoryBmagnificentlyBlistingBliliBlayerBlatinoB	katharineBjennaB
jacquelineBjacobiB	intellectBimpressionsBimpersonationB	imitatingBideologyBhealB	hardshipsBgruffBgrannyBgrandsonBgimmicksBgershwinBgentlyBfewerBexcessesBevolveBepitomeBenhancesB	emptinessBellisB	eliminateB	effectsbrBdrowningBdiscussionsBdiscernibleB	dillingerB
devastatedBdemiBdelonB	deliriousBcrummyBcornersB	concludesB
compromiseBcompetentlyB
companionsBcolBcockneyBclimateBcarrollB	carnosaurBburialBbulliesBbrynnerBbridesBboxesB	blackmailBbedsBbeauBbagsBbafflingBarkinBaptlyBannoysBanitaBanalyzeBaimlessB	agreementBadvisedBablyB2009B1964BzoomByetiBwitsBwiresBwinstonBwhodBvoyagerBvincenzoBvetBuniversallyBunhingedB
unemployedBturgidBtinBtimedBtideBtextbookBsunriseBstubbornBstonerBstampBspousesBspittingBspelledBsoylentBsopranoBsnippetsBsmashingBslotBslewBslaveryBsinatrasBshapesBshaftBseventhBserbianBschtickBsatisfactoryBsabotageBrushesBrolandBrkoBretainB	rehearsalBreeseB
recoveringBraeBrachaelBquigleyBproudlyBproposesB	promotionBpreteenBpremierBprecededB	practicesBponderBphrasesBoutlookBorientalB
oppressiveB
oppositionBnotwithstandingBnodsBninetiesBneglectB	misplacedBmarcelBlevyBleonardoBladysBkennyBirresistibleBintimacyB
infidelityBillustratedBholidaysBhispanicBheritageBhensonBhectorBharlowBhannibalBgroundedBgriffinBgladlyBfuturebrBfundsBfreelyB	footstepsBfinanceBfieryBfiascoBeyebrowsB
expressiveBexposingB	exchangesBexaminedBembarkBduncanBdownwardB	docudramaBdiegoBdeviousBdefinesBdasBdarwinBdarkestBdarcyBdaphneBdakotaBcustomsBcringedB	cowrittenBcoverageB
connectingBcompositionsB
collectingBcollarBcoalBchavezBcaroleBcarly
??
Const_5Const*
_output_shapes	
:?N*
dtype0	*??
value??B??	?N"??                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	       
      
      
      
      
      
      
      
      
      	
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
       
      !
      "
      #
      $
      %
      &
      '
      (
      )
      *
      +
      ,
      -
      .
      /
      0
      1
      2
      3
      4
      5
      6
      7
      8
      9
      :
      ;
      <
      =
      >
      ?
      @
      A
      B
      C
      D
      E
      F
      G
      H
      I
      J
      K
      L
      M
      N
      O
      P
      Q
      R
      S
      T
      U
      V
      W
      X
      Y
      Z
      [
      \
      ]
      ^
      _
      `
      a
      b
      c
      d
      e
      f
      g
      h
      i
      j
      k
      l
      m
      n
      o
      p
      q
      r
      s
      t
      u
      v
      w
      x
      y
      z
      {
      |
      }
      ~
      
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                                      	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?        !      !      !      !      !      !      !      !      !      	!      
!      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !       !      !!      "!      #!      $!      %!      &!      '!      (!      )!      *!      +!      ,!      -!      .!      /!      0!      1!      2!      3!      4!      5!      6!      7!      8!      9!      :!      ;!      <!      =!      >!      ?!      @!      A!      B!      C!      D!      E!      F!      G!      H!      I!      J!      K!      L!      M!      N!      O!      P!      Q!      R!      S!      T!      U!      V!      W!      X!      Y!      Z!      [!      \!      ]!      ^!      _!      `!      a!      b!      c!      d!      e!      f!      g!      h!      i!      j!      k!      l!      m!      n!      o!      p!      q!      r!      s!      t!      u!      v!      w!      x!      y!      z!      {!      |!      }!      ~!      !      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!       "      "      "      "      "      "      "      "      "      	"      
"      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "       "      !"      ""      #"      $"      %"      &"      '"      ("      )"      *"      +"      ,"      -"      ."      /"      0"      1"      2"      3"      4"      5"      6"      7"      8"      9"      :"      ;"      <"      ="      >"      ?"      @"      A"      B"      C"      D"      E"      F"      G"      H"      I"      J"      K"      L"      M"      N"      O"      P"      Q"      R"      S"      T"      U"      V"      W"      X"      Y"      Z"      ["      \"      ]"      ^"      _"      `"      a"      b"      c"      d"      e"      f"      g"      h"      i"      j"      k"      l"      m"      n"      o"      p"      q"      r"      s"      t"      u"      v"      w"      x"      y"      z"      {"      |"      }"      ~"      "      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"       #      #      #      #      #      #      #      #      #      	#      
#      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #       #      !#      "#      ##      $#      %#      &#      '#      (#      )#      *#      +#      ,#      -#      .#      /#      0#      1#      2#      3#      4#      5#      6#      7#      8#      9#      :#      ;#      <#      =#      >#      ?#      @#      A#      B#      C#      D#      E#      F#      G#      H#      I#      J#      K#      L#      M#      N#      O#      P#      Q#      R#      S#      T#      U#      V#      W#      X#      Y#      Z#      [#      \#      ]#      ^#      _#      `#      a#      b#      c#      d#      e#      f#      g#      h#      i#      j#      k#      l#      m#      n#      o#      p#      q#      r#      s#      t#      u#      v#      w#      x#      y#      z#      {#      |#      }#      ~#      #      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#       $      $      $      $      $      $      $      $      $      	$      
$      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $       $      !$      "$      #$      $$      %$      &$      '$      ($      )$      *$      +$      ,$      -$      .$      /$      0$      1$      2$      3$      4$      5$      6$      7$      8$      9$      :$      ;$      <$      =$      >$      ?$      @$      A$      B$      C$      D$      E$      F$      G$      H$      I$      J$      K$      L$      M$      N$      O$      P$      Q$      R$      S$      T$      U$      V$      W$      X$      Y$      Z$      [$      \$      ]$      ^$      _$      `$      a$      b$      c$      d$      e$      f$      g$      h$      i$      j$      k$      l$      m$      n$      o$      p$      q$      r$      s$      t$      u$      v$      w$      x$      y$      z$      {$      |$      }$      ~$      $      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$       %      %      %      %      %      %      %      %      %      	%      
%      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %       %      !%      "%      #%      $%      %%      &%      '%      (%      )%      *%      +%      ,%      -%      .%      /%      0%      1%      2%      3%      4%      5%      6%      7%      8%      9%      :%      ;%      <%      =%      >%      ?%      @%      A%      B%      C%      D%      E%      F%      G%      H%      I%      J%      K%      L%      M%      N%      O%      P%      Q%      R%      S%      T%      U%      V%      W%      X%      Y%      Z%      [%      \%      ]%      ^%      _%      `%      a%      b%      c%      d%      e%      f%      g%      h%      i%      j%      k%      l%      m%      n%      o%      p%      q%      r%      s%      t%      u%      v%      w%      x%      y%      z%      {%      |%      }%      ~%      %      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%       &      &      &      &      &      &      &      &      &      	&      
&      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &       &      !&      "&      #&      $&      %&      &&      '&      (&      )&      *&      +&      ,&      -&      .&      /&      0&      1&      2&      3&      4&      5&      6&      7&      8&      9&      :&      ;&      <&      =&      >&      ?&      @&      A&      B&      C&      D&      E&      F&      G&      H&      I&      J&      K&      L&      M&      N&      O&      P&      Q&      R&      S&      T&      U&      V&      W&      X&      Y&      Z&      [&      \&      ]&      ^&      _&      `&      a&      b&      c&      d&      e&      f&      g&      h&      i&      j&      k&      l&      m&      n&      o&      p&      q&      r&      s&      t&      u&      v&      w&      x&      y&      z&      {&      |&      }&      ~&      &      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&       '      '      '      '      '      '      '      '      '      	'      
'      '      '      '      '      '      
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_152371
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_152376
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?;
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?:
value?:B?: B?:
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
;
_lookup_layer
	keras_api
_adapt_function*
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
?
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses* 
?

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses*
?

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses*
?

8beta_1

9beta_2
	:decay
;learning_rate
<itermqmrms(mt)mu0mv1mwvxvyvz(v{)v|0v}1v~*
5
1
2
3
(4
)5
06
17*
5
0
1
2
(3
)4
05
16*
* 
?
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Bserving_default* 
7
Clookup_table
Dtoken_counts
E	keras_api*
* 
* 
hb
VARIABLE_VALUEembedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEconv1d/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv1d/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses* 
* 
* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

(0
)1*

(0
)1*
* 
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

00
11*

00
11*
* 
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*
* 
* 
KE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

_0
`1*
* 
* 
* 
R
a_initializer
b_create_resource
c_initialize
d_destroy_resource* 
?
e_create_resource
f_initialize
g_destroy_resourceJ
tableAlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	htotal
	icount
j	variables
k	keras_api*
H
	ltotal
	mcount
n
_fn_kwargs
o	variables
p	keras_api*
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

h0
i1*

j	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

l0
m1*

o	variables*
??
VARIABLE_VALUEAdam/embedding/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv1d/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv1d/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/embedding/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv1d/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv1d/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
.serving_default_Text_Vectorization_Layer_inputPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_1StatefulPartitionedCall.serving_default_Text_Vectorization_Layer_input
hash_tableConstConst_1Const_2embedding/embeddingsconv1d/kernelconv1d/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_152163
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1total/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp(Adam/conv1d/kernel/m/Read/ReadVariableOp&Adam/conv1d/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp(Adam/conv1d/kernel/v/Read/ReadVariableOp&Adam/conv1d/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst_6*-
Tin&
$2"		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_152503
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameembedding/embeddingsconv1d/kernelconv1d/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasbeta_1beta_2decaylearning_rate	Adam/iterMutableHashTabletotalcounttotal_1count_1Adam/embedding/embeddings/mAdam/conv1d/kernel/mAdam/conv1d/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/embedding/embeddings/vAdam/conv1d/kernel/vAdam/conv1d/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_152606??

?q
?
F__inference_sequential_layer_call_and_return_conditional_losses_151906"
text_vectorization_layer_inputU
Qtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_default_value	2
.text_vectorization_layer_string_lookup_equal_y5
1text_vectorization_layer_string_lookup_selectv2_t	#
embedding_151886:	?N#
conv1d_151889:@
conv1d_151891:@
dense_151895:@	
dense_151897:	 
dense_1_151900:	
dense_1_151902:
identity??DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2?conv1d/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!embedding/StatefulPartitionedCallx
$Text_Vectorization_Layer/StringLowerStringLowertext_vectorization_layer_input*#
_output_shapes
:??????????
+Text_Vectorization_Layer/StaticRegexReplaceStaticRegexReplace-Text_Vectorization_Layer/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite k
*Text_Vectorization_Layer/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
2Text_Vectorization_Layer/StringSplit/StringSplitV2StringSplitV24Text_Vectorization_Layer/StaticRegexReplace:output:03Text_Vectorization_Layer/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
8Text_Vectorization_Layer/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
:Text_Vectorization_Layer/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
:Text_Vectorization_Layer/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
2Text_Vectorization_Layer/StringSplit/strided_sliceStridedSlice<Text_Vectorization_Layer/StringSplit/StringSplitV2:indices:0AText_Vectorization_Layer/StringSplit/strided_slice/stack:output:0CText_Vectorization_Layer/StringSplit/strided_slice/stack_1:output:0CText_Vectorization_Layer/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
:Text_Vectorization_Layer/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<Text_Vectorization_Layer/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<Text_Vectorization_Layer/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4Text_Vectorization_Layer/StringSplit/strided_slice_1StridedSlice:Text_Vectorization_Layer/StringSplit/StringSplitV2:shape:0CText_Vectorization_Layer/StringSplit/strided_slice_1/stack:output:0EText_Vectorization_Layer/StringSplit/strided_slice_1/stack_1:output:0EText_Vectorization_Layer/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
[Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast;Text_Vectorization_Layer/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast=Text_Vectorization_Layer/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdnText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0nText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
iText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatermText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0rText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
dText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastkText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0pText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2lText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0nText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulhText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumaText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumaText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
hText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0pText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
bText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumoText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
fText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
bText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2oText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_table_handle;Text_Vectorization_Layer/StringSplit/StringSplitV2:values:0Rtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,Text_Vectorization_Layer/string_lookup/EqualEqual;Text_Vectorization_Layer/StringSplit/StringSplitV2:values:0.text_vectorization_layer_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
/Text_Vectorization_Layer/string_lookup/SelectV2SelectV20Text_Vectorization_Layer/string_lookup/Equal:z:01text_vectorization_layer_string_lookup_selectv2_tMText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/Text_Vectorization_Layer/string_lookup/IdentityIdentity8Text_Vectorization_Layer/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????w
5Text_Vectorization_Layer/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
-Text_Vectorization_Layer/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
<Text_Vectorization_Layer/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor6Text_Vectorization_Layer/RaggedToTensor/Const:output:08Text_Vectorization_Layer/string_lookup/Identity:output:0>Text_Vectorization_Layer/RaggedToTensor/default_value:output:0=Text_Vectorization_Layer/StringSplit/strided_slice_1:output:0;Text_Vectorization_Layer/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCallEText_Vectorization_Layer/RaggedToTensor/RaggedTensorToTensor:result:0embedding_151886*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_151490?
conv1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_151889conv1d_151891*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_151510?
(global_average_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_151424?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_151895dense_151897*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_151528?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_151900dense_1_151902*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_151545w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOpE^Text_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2^conv1d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2?
DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV22@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:c _
#
_output_shapes
:?????????
8
_user_specified_name Text_Vectorization_Layer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
E__inference_embedding_layer_call_and_return_conditional_losses_152227

inputs	*
embedding_lookup_152221:	?N
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_152221inputs*
Tindices0	**
_class 
loc:@embedding_lookup/152221*,
_output_shapes
:??????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/152221*,
_output_shapes
:???????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????x
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:??????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_conv1d_layer_call_fn_152236

inputs
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_151510t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_1_layer_call_fn_152292

inputs
unknown:	
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_151545o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
+
__inference_<lambda>_152376
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference__initializer_1523167
3key_value_init2702_lookuptableimportv2_table_handle/
+key_value_init2702_lookuptableimportv2_keys1
-key_value_init2702_lookuptableimportv2_values	
identity??&key_value_init2702/LookupTableImportV2?
&key_value_init2702/LookupTableImportV2LookupTableImportV23key_value_init2702_lookuptableimportv2_table_handle+key_value_init2702_lookuptableimportv2_keys-key_value_init2702_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: o
NoOpNoOp'^key_value_init2702/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?N:?N2P
&key_value_init2702/LookupTableImportV2&key_value_init2702/LookupTableImportV2:!

_output_shapes	
:?N:!

_output_shapes	
:?N
?
?
+__inference_sequential_layer_call_fn_151766"
text_vectorization_layer_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:@
	unknown_5:@
	unknown_6:@	
	unknown_7:	
	unknown_8:	
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_151714o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
#
_output_shapes
:?????????
8
_user_specified_name Text_Vectorization_Layer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_151424

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
݁
?
F__inference_sequential_layer_call_and_return_conditional_losses_152134

inputsU
Qtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_default_value	2
.text_vectorization_layer_string_lookup_equal_y5
1text_vectorization_layer_string_lookup_selectv2_t	4
!embedding_embedding_lookup_152100:	?NH
2conv1d_conv1d_expanddims_1_readvariableop_resource:@4
&conv1d_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@	3
%dense_biasadd_readvariableop_resource:	8
&dense_1_matmul_readvariableop_resource:	5
'dense_1_biasadd_readvariableop_resource:
identity??DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2?conv1d/BiasAdd/ReadVariableOp?)conv1d/Conv1D/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?embedding/embedding_lookup`
$Text_Vectorization_Layer/StringLowerStringLowerinputs*#
_output_shapes
:??????????
+Text_Vectorization_Layer/StaticRegexReplaceStaticRegexReplace-Text_Vectorization_Layer/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite k
*Text_Vectorization_Layer/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
2Text_Vectorization_Layer/StringSplit/StringSplitV2StringSplitV24Text_Vectorization_Layer/StaticRegexReplace:output:03Text_Vectorization_Layer/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
8Text_Vectorization_Layer/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
:Text_Vectorization_Layer/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
:Text_Vectorization_Layer/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
2Text_Vectorization_Layer/StringSplit/strided_sliceStridedSlice<Text_Vectorization_Layer/StringSplit/StringSplitV2:indices:0AText_Vectorization_Layer/StringSplit/strided_slice/stack:output:0CText_Vectorization_Layer/StringSplit/strided_slice/stack_1:output:0CText_Vectorization_Layer/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
:Text_Vectorization_Layer/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<Text_Vectorization_Layer/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<Text_Vectorization_Layer/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4Text_Vectorization_Layer/StringSplit/strided_slice_1StridedSlice:Text_Vectorization_Layer/StringSplit/StringSplitV2:shape:0CText_Vectorization_Layer/StringSplit/strided_slice_1/stack:output:0EText_Vectorization_Layer/StringSplit/strided_slice_1/stack_1:output:0EText_Vectorization_Layer/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
[Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast;Text_Vectorization_Layer/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast=Text_Vectorization_Layer/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdnText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0nText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
iText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatermText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0rText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
dText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastkText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0pText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2lText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0nText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulhText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumaText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumaText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
hText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0pText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
bText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumoText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
fText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
bText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2oText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_table_handle;Text_Vectorization_Layer/StringSplit/StringSplitV2:values:0Rtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,Text_Vectorization_Layer/string_lookup/EqualEqual;Text_Vectorization_Layer/StringSplit/StringSplitV2:values:0.text_vectorization_layer_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
/Text_Vectorization_Layer/string_lookup/SelectV2SelectV20Text_Vectorization_Layer/string_lookup/Equal:z:01text_vectorization_layer_string_lookup_selectv2_tMText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/Text_Vectorization_Layer/string_lookup/IdentityIdentity8Text_Vectorization_Layer/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????w
5Text_Vectorization_Layer/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
-Text_Vectorization_Layer/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
<Text_Vectorization_Layer/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor6Text_Vectorization_Layer/RaggedToTensor/Const:output:08Text_Vectorization_Layer/string_lookup/Identity:output:0>Text_Vectorization_Layer/RaggedToTensor/default_value:output:0=Text_Vectorization_Layer/StringSplit/strided_slice_1:output:0;Text_Vectorization_Layer/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_152100EText_Vectorization_Layer/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*4
_class*
(&loc:@embedding/embedding_lookup/152100*,
_output_shapes
:??????????*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/152100*,
_output_shapes
:???????????
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:???????????
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@?
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingVALID*
strides
?
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims

??????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d/MeanMeanconv1d/Relu:activations:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@	*
dtype0?
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOpE^Text_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2?
DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV22>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_<lambda>_1523717
3key_value_init2702_lookuptableimportv2_table_handle/
+key_value_init2702_lookuptableimportv2_keys1
-key_value_init2702_lookuptableimportv2_values	
identity??&key_value_init2702/LookupTableImportV2?
&key_value_init2702/LookupTableImportV2LookupTableImportV23key_value_init2702_lookuptableimportv2_table_handle+key_value_init2702_lookuptableimportv2_keys-key_value_init2702_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: o
NoOpNoOp'^key_value_init2702/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?N:?N2P
&key_value_init2702/LookupTableImportV2&key_value_init2702/LookupTableImportV2:!

_output_shapes	
:?N:!

_output_shapes	
:?N
?
?
B__inference_conv1d_layer_call_and_return_conditional_losses_151510

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:???????????
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims

?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????@f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????@?
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
U
9__inference_global_average_pooling1d_layer_call_fn_152257

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_151424i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?

?
C__inference_dense_1_layer_call_and_return_conditional_losses_151545

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
݁
?
F__inference_sequential_layer_call_and_return_conditional_losses_152050

inputsU
Qtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_default_value	2
.text_vectorization_layer_string_lookup_equal_y5
1text_vectorization_layer_string_lookup_selectv2_t	4
!embedding_embedding_lookup_152016:	?NH
2conv1d_conv1d_expanddims_1_readvariableop_resource:@4
&conv1d_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@	3
%dense_biasadd_readvariableop_resource:	8
&dense_1_matmul_readvariableop_resource:	5
'dense_1_biasadd_readvariableop_resource:
identity??DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2?conv1d/BiasAdd/ReadVariableOp?)conv1d/Conv1D/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?embedding/embedding_lookup`
$Text_Vectorization_Layer/StringLowerStringLowerinputs*#
_output_shapes
:??????????
+Text_Vectorization_Layer/StaticRegexReplaceStaticRegexReplace-Text_Vectorization_Layer/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite k
*Text_Vectorization_Layer/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
2Text_Vectorization_Layer/StringSplit/StringSplitV2StringSplitV24Text_Vectorization_Layer/StaticRegexReplace:output:03Text_Vectorization_Layer/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
8Text_Vectorization_Layer/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
:Text_Vectorization_Layer/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
:Text_Vectorization_Layer/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
2Text_Vectorization_Layer/StringSplit/strided_sliceStridedSlice<Text_Vectorization_Layer/StringSplit/StringSplitV2:indices:0AText_Vectorization_Layer/StringSplit/strided_slice/stack:output:0CText_Vectorization_Layer/StringSplit/strided_slice/stack_1:output:0CText_Vectorization_Layer/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
:Text_Vectorization_Layer/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<Text_Vectorization_Layer/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<Text_Vectorization_Layer/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4Text_Vectorization_Layer/StringSplit/strided_slice_1StridedSlice:Text_Vectorization_Layer/StringSplit/StringSplitV2:shape:0CText_Vectorization_Layer/StringSplit/strided_slice_1/stack:output:0EText_Vectorization_Layer/StringSplit/strided_slice_1/stack_1:output:0EText_Vectorization_Layer/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
[Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast;Text_Vectorization_Layer/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast=Text_Vectorization_Layer/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdnText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0nText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
iText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatermText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0rText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
dText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastkText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0pText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2lText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0nText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulhText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumaText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumaText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
hText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0pText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
bText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumoText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
fText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
bText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2oText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_table_handle;Text_Vectorization_Layer/StringSplit/StringSplitV2:values:0Rtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,Text_Vectorization_Layer/string_lookup/EqualEqual;Text_Vectorization_Layer/StringSplit/StringSplitV2:values:0.text_vectorization_layer_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
/Text_Vectorization_Layer/string_lookup/SelectV2SelectV20Text_Vectorization_Layer/string_lookup/Equal:z:01text_vectorization_layer_string_lookup_selectv2_tMText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/Text_Vectorization_Layer/string_lookup/IdentityIdentity8Text_Vectorization_Layer/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????w
5Text_Vectorization_Layer/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
-Text_Vectorization_Layer/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
<Text_Vectorization_Layer/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor6Text_Vectorization_Layer/RaggedToTensor/Const:output:08Text_Vectorization_Layer/string_lookup/Identity:output:0>Text_Vectorization_Layer/RaggedToTensor/default_value:output:0=Text_Vectorization_Layer/StringSplit/strided_slice_1:output:0;Text_Vectorization_Layer/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_152016EText_Vectorization_Layer/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*4
_class*
(&loc:@embedding/embedding_lookup/152016*,
_output_shapes
:??????????*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/152016*,
_output_shapes
:???????????
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:???????????
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@?
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingVALID*
strides
?
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims

??????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d/MeanMeanconv1d/Relu:activations:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@	*
dtype0?
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOpE^Text_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2?
DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV22>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_save_fn_152355
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::K
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: O
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?p
?
F__inference_sequential_layer_call_and_return_conditional_losses_151552

inputsU
Qtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_default_value	2
.text_vectorization_layer_string_lookup_equal_y5
1text_vectorization_layer_string_lookup_selectv2_t	#
embedding_151491:	?N#
conv1d_151511:@
conv1d_151513:@
dense_151529:@	
dense_151531:	 
dense_1_151546:	
dense_1_151548:
identity??DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2?conv1d/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!embedding/StatefulPartitionedCall`
$Text_Vectorization_Layer/StringLowerStringLowerinputs*#
_output_shapes
:??????????
+Text_Vectorization_Layer/StaticRegexReplaceStaticRegexReplace-Text_Vectorization_Layer/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite k
*Text_Vectorization_Layer/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
2Text_Vectorization_Layer/StringSplit/StringSplitV2StringSplitV24Text_Vectorization_Layer/StaticRegexReplace:output:03Text_Vectorization_Layer/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
8Text_Vectorization_Layer/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
:Text_Vectorization_Layer/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
:Text_Vectorization_Layer/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
2Text_Vectorization_Layer/StringSplit/strided_sliceStridedSlice<Text_Vectorization_Layer/StringSplit/StringSplitV2:indices:0AText_Vectorization_Layer/StringSplit/strided_slice/stack:output:0CText_Vectorization_Layer/StringSplit/strided_slice/stack_1:output:0CText_Vectorization_Layer/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
:Text_Vectorization_Layer/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<Text_Vectorization_Layer/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<Text_Vectorization_Layer/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4Text_Vectorization_Layer/StringSplit/strided_slice_1StridedSlice:Text_Vectorization_Layer/StringSplit/StringSplitV2:shape:0CText_Vectorization_Layer/StringSplit/strided_slice_1/stack:output:0EText_Vectorization_Layer/StringSplit/strided_slice_1/stack_1:output:0EText_Vectorization_Layer/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
[Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast;Text_Vectorization_Layer/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast=Text_Vectorization_Layer/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdnText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0nText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
iText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatermText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0rText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
dText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastkText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0pText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2lText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0nText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulhText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumaText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumaText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
hText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0pText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
bText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumoText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
fText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
bText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2oText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_table_handle;Text_Vectorization_Layer/StringSplit/StringSplitV2:values:0Rtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,Text_Vectorization_Layer/string_lookup/EqualEqual;Text_Vectorization_Layer/StringSplit/StringSplitV2:values:0.text_vectorization_layer_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
/Text_Vectorization_Layer/string_lookup/SelectV2SelectV20Text_Vectorization_Layer/string_lookup/Equal:z:01text_vectorization_layer_string_lookup_selectv2_tMText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/Text_Vectorization_Layer/string_lookup/IdentityIdentity8Text_Vectorization_Layer/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????w
5Text_Vectorization_Layer/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
-Text_Vectorization_Layer/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
<Text_Vectorization_Layer/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor6Text_Vectorization_Layer/RaggedToTensor/Const:output:08Text_Vectorization_Layer/string_lookup/Identity:output:0>Text_Vectorization_Layer/RaggedToTensor/default_value:output:0=Text_Vectorization_Layer/StringSplit/strided_slice_1:output:0;Text_Vectorization_Layer/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCallEText_Vectorization_Layer/RaggedToTensor/RaggedTensorToTensor:result:0embedding_151491*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_151490?
conv1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_151511conv1d_151513*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_151510?
(global_average_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_151424?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_151529dense_151531*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_151528?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_151546dense_1_151548*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_151545w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOpE^Text_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2^conv1d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2?
DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV22@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
+__inference_sequential_layer_call_fn_151966

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:@
	unknown_5:@
	unknown_6:@	
	unknown_7:	
	unknown_8:	
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_151714o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
-
__inference__destroyer_152321
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?C
?
__inference_adapt_step_152211
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2]
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:??????????
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
?q
?
F__inference_sequential_layer_call_and_return_conditional_losses_151836"
text_vectorization_layer_inputU
Qtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_default_value	2
.text_vectorization_layer_string_lookup_equal_y5
1text_vectorization_layer_string_lookup_selectv2_t	#
embedding_151816:	?N#
conv1d_151819:@
conv1d_151821:@
dense_151825:@	
dense_151827:	 
dense_1_151830:	
dense_1_151832:
identity??DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2?conv1d/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!embedding/StatefulPartitionedCallx
$Text_Vectorization_Layer/StringLowerStringLowertext_vectorization_layer_input*#
_output_shapes
:??????????
+Text_Vectorization_Layer/StaticRegexReplaceStaticRegexReplace-Text_Vectorization_Layer/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite k
*Text_Vectorization_Layer/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
2Text_Vectorization_Layer/StringSplit/StringSplitV2StringSplitV24Text_Vectorization_Layer/StaticRegexReplace:output:03Text_Vectorization_Layer/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
8Text_Vectorization_Layer/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
:Text_Vectorization_Layer/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
:Text_Vectorization_Layer/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
2Text_Vectorization_Layer/StringSplit/strided_sliceStridedSlice<Text_Vectorization_Layer/StringSplit/StringSplitV2:indices:0AText_Vectorization_Layer/StringSplit/strided_slice/stack:output:0CText_Vectorization_Layer/StringSplit/strided_slice/stack_1:output:0CText_Vectorization_Layer/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
:Text_Vectorization_Layer/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<Text_Vectorization_Layer/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<Text_Vectorization_Layer/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4Text_Vectorization_Layer/StringSplit/strided_slice_1StridedSlice:Text_Vectorization_Layer/StringSplit/StringSplitV2:shape:0CText_Vectorization_Layer/StringSplit/strided_slice_1/stack:output:0EText_Vectorization_Layer/StringSplit/strided_slice_1/stack_1:output:0EText_Vectorization_Layer/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
[Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast;Text_Vectorization_Layer/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast=Text_Vectorization_Layer/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdnText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0nText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
iText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatermText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0rText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
dText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastkText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0pText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2lText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0nText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulhText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumaText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumaText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
hText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0pText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
bText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumoText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
fText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
bText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2oText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_table_handle;Text_Vectorization_Layer/StringSplit/StringSplitV2:values:0Rtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,Text_Vectorization_Layer/string_lookup/EqualEqual;Text_Vectorization_Layer/StringSplit/StringSplitV2:values:0.text_vectorization_layer_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
/Text_Vectorization_Layer/string_lookup/SelectV2SelectV20Text_Vectorization_Layer/string_lookup/Equal:z:01text_vectorization_layer_string_lookup_selectv2_tMText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/Text_Vectorization_Layer/string_lookup/IdentityIdentity8Text_Vectorization_Layer/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????w
5Text_Vectorization_Layer/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
-Text_Vectorization_Layer/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
<Text_Vectorization_Layer/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor6Text_Vectorization_Layer/RaggedToTensor/Const:output:08Text_Vectorization_Layer/string_lookup/Identity:output:0>Text_Vectorization_Layer/RaggedToTensor/default_value:output:0=Text_Vectorization_Layer/StringSplit/strided_slice_1:output:0;Text_Vectorization_Layer/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCallEText_Vectorization_Layer/RaggedToTensor/RaggedTensorToTensor:result:0embedding_151816*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_151490?
conv1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_151819conv1d_151821*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_151510?
(global_average_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_151424?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_151825dense_151827*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_151528?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_151830dense_1_151832*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_151545w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOpE^Text_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2^conv1d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2?
DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV22@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:c _
#
_output_shapes
:?????????
8
_user_specified_name Text_Vectorization_Layer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
&__inference_dense_layer_call_fn_152272

inputs
unknown:@	
	unknown_0:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_151528o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
G
__inference__creator_152326
identity: ??MutableHashTable~
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_103*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?
?
E__inference_embedding_layer_call_and_return_conditional_losses_151490

inputs	*
embedding_lookup_151484:	?N
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_151484inputs*
Tindices0	**
_class 
loc:@embedding_lookup/151484*,
_output_shapes
:??????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/151484*,
_output_shapes
:???????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????x
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:??????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_restore_fn_152363
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?p
?
F__inference_sequential_layer_call_and_return_conditional_losses_151714

inputsU
Qtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_default_value	2
.text_vectorization_layer_string_lookup_equal_y5
1text_vectorization_layer_string_lookup_selectv2_t	#
embedding_151694:	?N#
conv1d_151697:@
conv1d_151699:@
dense_151703:@	
dense_151705:	 
dense_1_151708:	
dense_1_151710:
identity??DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2?conv1d/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!embedding/StatefulPartitionedCall`
$Text_Vectorization_Layer/StringLowerStringLowerinputs*#
_output_shapes
:??????????
+Text_Vectorization_Layer/StaticRegexReplaceStaticRegexReplace-Text_Vectorization_Layer/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite k
*Text_Vectorization_Layer/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
2Text_Vectorization_Layer/StringSplit/StringSplitV2StringSplitV24Text_Vectorization_Layer/StaticRegexReplace:output:03Text_Vectorization_Layer/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
8Text_Vectorization_Layer/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
:Text_Vectorization_Layer/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
:Text_Vectorization_Layer/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
2Text_Vectorization_Layer/StringSplit/strided_sliceStridedSlice<Text_Vectorization_Layer/StringSplit/StringSplitV2:indices:0AText_Vectorization_Layer/StringSplit/strided_slice/stack:output:0CText_Vectorization_Layer/StringSplit/strided_slice/stack_1:output:0CText_Vectorization_Layer/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
:Text_Vectorization_Layer/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<Text_Vectorization_Layer/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<Text_Vectorization_Layer/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4Text_Vectorization_Layer/StringSplit/strided_slice_1StridedSlice:Text_Vectorization_Layer/StringSplit/StringSplitV2:shape:0CText_Vectorization_Layer/StringSplit/strided_slice_1/stack:output:0EText_Vectorization_Layer/StringSplit/strided_slice_1/stack_1:output:0EText_Vectorization_Layer/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
[Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast;Text_Vectorization_Layer/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast=Text_Vectorization_Layer/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdnText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0nText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
iText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatermText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0rText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
dText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastkText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0pText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
eText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2lText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0nText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulhText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumaText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumaText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
gText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
hText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount_Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0pText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
bText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumoText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
fText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
bText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
]Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2oText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0cText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0kText_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_table_handle;Text_Vectorization_Layer/StringSplit/StringSplitV2:values:0Rtext_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,Text_Vectorization_Layer/string_lookup/EqualEqual;Text_Vectorization_Layer/StringSplit/StringSplitV2:values:0.text_vectorization_layer_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
/Text_Vectorization_Layer/string_lookup/SelectV2SelectV20Text_Vectorization_Layer/string_lookup/Equal:z:01text_vectorization_layer_string_lookup_selectv2_tMText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/Text_Vectorization_Layer/string_lookup/IdentityIdentity8Text_Vectorization_Layer/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????w
5Text_Vectorization_Layer/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
-Text_Vectorization_Layer/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
<Text_Vectorization_Layer/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor6Text_Vectorization_Layer/RaggedToTensor/Const:output:08Text_Vectorization_Layer/string_lookup/Identity:output:0>Text_Vectorization_Layer/RaggedToTensor/default_value:output:0=Text_Vectorization_Layer/StringSplit/strided_slice_1:output:0;Text_Vectorization_Layer/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCallEText_Vectorization_Layer/RaggedToTensor/RaggedTensorToTensor:result:0embedding_151694*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_151490?
conv1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_151697conv1d_151699*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_151510?
(global_average_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_151424?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_151703dense_151705*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_151528?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_151708dense_1_151710*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_151545w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOpE^Text_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2^conv1d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2?
DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2DText_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV22@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?D
?
__inference__traced_save_152503
file_prefix3
/savev2_embedding_embeddings_read_readvariableop,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	J
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop3
/savev2_adam_conv1d_kernel_m_read_readvariableop1
-savev2_adam_conv1d_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop3
/savev2_adam_conv1d_kernel_v_read_readvariableop1
-savev2_adam_conv1d_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const_6

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1 savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop/savev2_adam_conv1d_kernel_m_read_readvariableop-savev2_adam_conv1d_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop/savev2_adam_conv1d_kernel_v_read_readvariableop-savev2_adam_conv1d_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?N:@:@:@	:	:	:: : : : : ::: : : : :	?N:@:@:@	:	:	::	?N:@:@:@	:	:	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?N:($
"
_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@	: 

_output_shapes
:	:$ 

_output_shapes

:	: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?N:($
"
_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@	: 

_output_shapes
:	:$ 

_output_shapes

:	: 

_output_shapes
::%!

_output_shapes
:	?N:($
"
_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@	: 

_output_shapes
:	:$ 

_output_shapes

:	:  

_output_shapes
::!

_output_shapes
: 
?

?
C__inference_dense_1_layer_call_and_return_conditional_losses_152303

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_152163"
text_vectorization_layer_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:@
	unknown_5:@
	unknown_6:@	
	unknown_7:	
	unknown_8:	
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_151414o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
#
_output_shapes
:?????????
8
_user_specified_name Text_Vectorization_Layer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
A__inference_dense_layer_call_and_return_conditional_losses_152283

inputs0
matmul_readvariableop_resource:@	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????	a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
/
__inference__initializer_152331
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?~
?
"__inference__traced_restore_152606
file_prefix8
%assignvariableop_embedding_embeddings:	?N6
 assignvariableop_1_conv1d_kernel:@,
assignvariableop_2_conv1d_bias:@1
assignvariableop_3_dense_kernel:@	+
assignvariableop_4_dense_bias:	3
!assignvariableop_5_dense_1_kernel:	-
assignvariableop_6_dense_1_bias:#
assignvariableop_7_beta_1: #
assignvariableop_8_beta_2: "
assignvariableop_9_decay: +
!assignvariableop_10_learning_rate: '
assignvariableop_11_adam_iter:	 M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: #
assignvariableop_12_total: #
assignvariableop_13_count: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: B
/assignvariableop_16_adam_embedding_embeddings_m:	?N>
(assignvariableop_17_adam_conv1d_kernel_m:@4
&assignvariableop_18_adam_conv1d_bias_m:@9
'assignvariableop_19_adam_dense_kernel_m:@	3
%assignvariableop_20_adam_dense_bias_m:	;
)assignvariableop_21_adam_dense_1_kernel_m:	5
'assignvariableop_22_adam_dense_1_bias_m:B
/assignvariableop_23_adam_embedding_embeddings_v:	?N>
(assignvariableop_24_adam_conv1d_kernel_v:@4
&assignvariableop_25_adam_conv1d_bias_v:@9
'assignvariableop_26_adam_dense_kernel_v:@	3
%assignvariableop_27_adam_dense_bias_v:	;
)assignvariableop_28_adam_dense_1_kernel_v:	5
'assignvariableop_29_adam_dense_1_bias_v:
identity_31??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?2MutableHashTable_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::*/
dtypes%
#2!		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv1d_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_1_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_1_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_iterIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:12RestoreV2:tensors:13*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 _
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp/assignvariableop_16_adam_embedding_embeddings_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_conv1d_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_conv1d_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_dense_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp/assignvariableop_23_adam_embedding_embeddings_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv1d_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp&assignvariableop_25_adam_conv1d_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_dense_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_1_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_1_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
??
?

!__inference__wrapped_model_151414"
text_vectorization_layer_input`
\sequential_text_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_table_handlea
]sequential_text_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_default_value	=
9sequential_text_vectorization_layer_string_lookup_equal_y@
<sequential_text_vectorization_layer_string_lookup_selectv2_t	?
,sequential_embedding_embedding_lookup_151380:	?NS
=sequential_conv1d_conv1d_expanddims_1_readvariableop_resource:@?
1sequential_conv1d_biasadd_readvariableop_resource:@A
/sequential_dense_matmul_readvariableop_resource:@	>
0sequential_dense_biasadd_readvariableop_resource:	C
1sequential_dense_1_matmul_readvariableop_resource:	@
2sequential_dense_1_biasadd_readvariableop_resource:
identity??Osequential/Text_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2?(sequential/conv1d/BiasAdd/ReadVariableOp?4sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?%sequential/embedding/embedding_lookup?
/sequential/Text_Vectorization_Layer/StringLowerStringLowertext_vectorization_layer_input*#
_output_shapes
:??????????
6sequential/Text_Vectorization_Layer/StaticRegexReplaceStaticRegexReplace8sequential/Text_Vectorization_Layer/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite v
5sequential/Text_Vectorization_Layer/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
=sequential/Text_Vectorization_Layer/StringSplit/StringSplitV2StringSplitV2?sequential/Text_Vectorization_Layer/StaticRegexReplace:output:0>sequential/Text_Vectorization_Layer/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
Csequential/Text_Vectorization_Layer/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Esequential/Text_Vectorization_Layer/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Esequential/Text_Vectorization_Layer/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
=sequential/Text_Vectorization_Layer/StringSplit/strided_sliceStridedSliceGsequential/Text_Vectorization_Layer/StringSplit/StringSplitV2:indices:0Lsequential/Text_Vectorization_Layer/StringSplit/strided_slice/stack:output:0Nsequential/Text_Vectorization_Layer/StringSplit/strided_slice/stack_1:output:0Nsequential/Text_Vectorization_Layer/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Esequential/Text_Vectorization_Layer/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Gsequential/Text_Vectorization_Layer/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Gsequential/Text_Vectorization_Layer/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
?sequential/Text_Vectorization_Layer/StringSplit/strided_slice_1StridedSliceEsequential/Text_Vectorization_Layer/StringSplit/StringSplitV2:shape:0Nsequential/Text_Vectorization_Layer/StringSplit/strided_slice_1/stack:output:0Psequential/Text_Vectorization_Layer/StringSplit/strided_slice_1/stack_1:output:0Psequential/Text_Vectorization_Layer/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
fsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastFsequential/Text_Vectorization_Layer/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
hsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastHsequential/Text_Vectorization_Layer/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
psequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapejsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
psequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
osequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdysequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ysequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
tsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
rsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterxsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0}sequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
osequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastvsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
rsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
nsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxjsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0{sequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
psequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
nsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2wsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ysequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
nsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulssequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0rsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
rsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumlsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0rsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
rsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumlsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0vsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
rsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
ssequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountjsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0vsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0{sequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
msequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
hsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumzsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0vsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
qsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
msequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
hsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2zsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0nsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0vsequential/Text_Vectorization_Layer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Osequential/Text_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2\sequential_text_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_table_handleFsequential/Text_Vectorization_Layer/StringSplit/StringSplitV2:values:0]sequential_text_vectorization_layer_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
7sequential/Text_Vectorization_Layer/string_lookup/EqualEqualFsequential/Text_Vectorization_Layer/StringSplit/StringSplitV2:values:09sequential_text_vectorization_layer_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
:sequential/Text_Vectorization_Layer/string_lookup/SelectV2SelectV2;sequential/Text_Vectorization_Layer/string_lookup/Equal:z:0<sequential_text_vectorization_layer_string_lookup_selectv2_tXsequential/Text_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
:sequential/Text_Vectorization_Layer/string_lookup/IdentityIdentityCsequential/Text_Vectorization_Layer/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
@sequential/Text_Vectorization_Layer/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
8sequential/Text_Vectorization_Layer/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
Gsequential/Text_Vectorization_Layer/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorAsequential/Text_Vectorization_Layer/RaggedToTensor/Const:output:0Csequential/Text_Vectorization_Layer/string_lookup/Identity:output:0Isequential/Text_Vectorization_Layer/RaggedToTensor/default_value:output:0Hsequential/Text_Vectorization_Layer/StringSplit/strided_slice_1:output:0Fsequential/Text_Vectorization_Layer/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
%sequential/embedding/embedding_lookupResourceGather,sequential_embedding_embedding_lookup_151380Psequential/Text_Vectorization_Layer/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*?
_class5
31loc:@sequential/embedding/embedding_lookup/151380*,
_output_shapes
:??????????*
dtype0?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*?
_class5
31loc:@sequential/embedding/embedding_lookup/151380*,
_output_shapes
:???????????
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????r
'sequential/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#sequential/conv1d/Conv1D/ExpandDims
ExpandDims9sequential/embedding/embedding_lookup/Identity_1:output:00sequential/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:???????????
4sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp=sequential_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0k
)sequential/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/conv1d/Conv1D/ExpandDims_1
ExpandDims<sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:02sequential/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@?
sequential/conv1d/Conv1DConv2D,sequential/conv1d/Conv1D/ExpandDims:output:0.sequential/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingVALID*
strides
?
 sequential/conv1d/Conv1D/SqueezeSqueeze!sequential/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims

??????????
(sequential/conv1d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential/conv1d/BiasAddBiasAdd)sequential/conv1d/Conv1D/Squeeze:output:00sequential/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@y
sequential/conv1d/ReluRelu"sequential/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@|
:sequential/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
(sequential/global_average_pooling1d/MeanMean$sequential/conv1d/Relu:activations:0Csequential/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@	*
dtype0?
sequential/dense/MatMulMatMul1sequential/global_average_pooling1d/Mean:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0?
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
sequential/dense_1/SigmoidSigmoid#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????m
IdentityIdentitysequential/dense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOpP^sequential/Text_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2)^sequential/conv1d/BiasAdd/ReadVariableOp5^sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2?
Osequential/Text_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV2Osequential/Text_Vectorization_Layer/string_lookup/None_Lookup/LookupTableFindV22T
(sequential/conv1d/BiasAdd/ReadVariableOp(sequential/conv1d/BiasAdd/ReadVariableOp2l
4sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp4sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup:c _
#
_output_shapes
:?????????
8
_user_specified_name Text_Vectorization_Layer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_152263

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
;
__inference__creator_152308
identity??
hash_tablel

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name2703*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?

*__inference_embedding_layer_call_fn_152218

inputs	
unknown:	?N
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_151490t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
A__inference_dense_layer_call_and_return_conditional_losses_151528

inputs0
matmul_readvariableop_resource:@	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????	a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
B__inference_conv1d_layer_call_and_return_conditional_losses_152252

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:???????????
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims

?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????@f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????@?
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
+__inference_sequential_layer_call_fn_151939

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:@
	unknown_5:@
	unknown_6:@	
	unknown_7:	
	unknown_8:	
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_151552o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_sequential_layer_call_fn_151577"
text_vectorization_layer_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:@
	unknown_5:@
	unknown_6:@	
	unknown_7:	
	unknown_8:	
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_layer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_151552o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
#
_output_shapes
:?????????
8
_user_specified_name Text_Vectorization_Layer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
-
__inference__destroyer_152336
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes "?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
e
Text_Vectorization_Layer_inputC
0serving_default_Text_Vectorization_Layer_input:0?????????=
dense_12
StatefulPartitionedCall_1:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
P
_lookup_layer
	keras_api
_adapt_function"
_tf_keras_layer
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
?
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
?

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
?

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
?

8beta_1

9beta_2
	:decay
;learning_rate
<itermqmrms(mt)mu0mv1mwvxvyvz(v{)v|0v}1v~"
	optimizer
Q
1
2
3
(4
)5
06
17"
trackable_list_wrapper
Q
0
1
2
(3
)4
05
16"
trackable_list_wrapper
 "
trackable_list_wrapper
?
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_sequential_layer_call_fn_151577
+__inference_sequential_layer_call_fn_151939
+__inference_sequential_layer_call_fn_151966
+__inference_sequential_layer_call_fn_151766?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_sequential_layer_call_and_return_conditional_losses_152050
F__inference_sequential_layer_call_and_return_conditional_losses_152134
F__inference_sequential_layer_call_and_return_conditional_losses_151836
F__inference_sequential_layer_call_and_return_conditional_losses_151906?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_151414Text_Vectorization_Layer_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
Bserving_default"
signature_map
L
Clookup_table
Dtoken_counts
E	keras_api"
_tf_keras_layer
"
_generic_user_object
?2?
__inference_adapt_step_152211?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
':%	?N2embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_embedding_layer_call_fn_152218?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_embedding_layer_call_and_return_conditional_losses_152227?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
#:!@2conv1d/kernel
:@2conv1d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_conv1d_layer_call_fn_152236?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv1d_layer_call_and_return_conditional_losses_152252?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
?2?
9__inference_global_average_pooling1d_layer_call_fn_152257?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_152263?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:@	2dense/kernel
:	2
dense/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_dense_layer_call_fn_152272?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_layer_call_and_return_conditional_losses_152283?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 :	2dense_1/kernel
:2dense_1/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_dense_1_layer_call_fn_152292?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_1_layer_call_and_return_conditional_losses_152303?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
$__inference_signature_wrapper_152163Text_Vectorization_Layer_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
j
a_initializer
b_create_resource
c_initialize
d_destroy_resourceR jCustom.StaticHashTable
P
e_create_resource
f_initialize
g_destroy_resourceR Z
table?
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	htotal
	icount
j	variables
k	keras_api"
_tf_keras_metric
^
	ltotal
	mcount
n
_fn_kwargs
o	variables
p	keras_api"
_tf_keras_metric
"
_generic_user_object
?2?
__inference__creator_152308?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_152316?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_152321?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_152326?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_152331?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_152336?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
:  (2total
:  (2count
.
h0
i1"
trackable_list_wrapper
-
j	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
l0
m1"
trackable_list_wrapper
-
o	variables"
_generic_user_object
,:*	?N2Adam/embedding/embeddings/m
(:&@2Adam/conv1d/kernel/m
:@2Adam/conv1d/bias/m
#:!@	2Adam/dense/kernel/m
:	2Adam/dense/bias/m
%:#	2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
,:*	?N2Adam/embedding/embeddings/v
(:&@2Adam/conv1d/kernel/v
:@2Adam/conv1d/bias/v
#:!@	2Adam/dense/kernel/v
:	2Adam/dense/bias/v
%:#	2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
?B?
__inference_save_fn_152355checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_152363restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_57
__inference__creator_152308?

? 
? "? 7
__inference__creator_152326?

? 
? "? 9
__inference__destroyer_152321?

? 
? "? 9
__inference__destroyer_152336?

? 
? "? B
__inference__initializer_152316C???

? 
? "? ;
__inference__initializer_152331?

? 
? "? ?
!__inference__wrapped_model_151414?C???()01C?@
9?6
4?1
Text_Vectorization_Layer_input?????????
? "1?.
,
dense_1!?
dense_1?????????k
__inference_adapt_step_152211JD???<
5?2
0?-?
??????????IteratorSpec 
? "
 ?
B__inference_conv1d_layer_call_and_return_conditional_losses_152252f4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????@
? ?
'__inference_conv1d_layer_call_fn_152236Y4?1
*?'
%?"
inputs??????????
? "???????????@?
C__inference_dense_1_layer_call_and_return_conditional_losses_152303\01/?,
%?"
 ?
inputs?????????	
? "%?"
?
0?????????
? {
(__inference_dense_1_layer_call_fn_152292O01/?,
%?"
 ?
inputs?????????	
? "???????????
A__inference_dense_layer_call_and_return_conditional_losses_152283\()/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????	
? y
&__inference_dense_layer_call_fn_152272O()/?,
%?"
 ?
inputs?????????@
? "??????????	?
E__inference_embedding_layer_call_and_return_conditional_losses_152227a0?-
&?#
!?
inputs??????????	
? "*?'
 ?
0??????????
? ?
*__inference_embedding_layer_call_fn_152218T0?-
&?#
!?
inputs??????????	
? "????????????
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_152263{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
9__inference_global_average_pooling1d_layer_call_fn_152257nI?F
??<
6?3
inputs'???????????????????????????

 
? "!???????????????????z
__inference_restore_fn_152363YDK?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_152355?D&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
F__inference_sequential_layer_call_and_return_conditional_losses_151836?C???()01K?H
A?>
4?1
Text_Vectorization_Layer_input?????????
p 

 
? "%?"
?
0?????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_151906?C???()01K?H
A?>
4?1
Text_Vectorization_Layer_input?????????
p

 
? "%?"
?
0?????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_152050lC???()013?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_152134lC???()013?0
)?&
?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
+__inference_sequential_layer_call_fn_151577wC???()01K?H
A?>
4?1
Text_Vectorization_Layer_input?????????
p 

 
? "???????????
+__inference_sequential_layer_call_fn_151766wC???()01K?H
A?>
4?1
Text_Vectorization_Layer_input?????????
p

 
? "???????????
+__inference_sequential_layer_call_fn_151939_C???()013?0
)?&
?
inputs?????????
p 

 
? "???????????
+__inference_sequential_layer_call_fn_151966_C???()013?0
)?&
?
inputs?????????
p

 
? "???????????
$__inference_signature_wrapper_152163?C???()01e?b
? 
[?X
V
Text_Vectorization_Layer_input4?1
Text_Vectorization_Layer_input?????????"1?.
,
dense_1!?
dense_1?????????