# 500 image-classification questions for dermal lesion images

Derived from the dermal-lesion assessment reference and its source table (131 sources). Each section lists the sources whose criteria, vocabularies, or findings ground its questions; author–year keys match the `authors`/`year` columns of `dermal_lesion_sources.csv`. Questions are numbered continuously so each can serve as a distinct classification feature.

## Image modality and acquisition (25)

*Grounding sources: Katragadda 2017; Kittler 2016; Pacheco 2020; Ghosh 2008; Cerminara 2023; Dinnes 2018d*

- **Q001.** Is this a clinical (macroscopic) photograph, a dermoscopic image, or a different imaging modality altogether?
- **Q002.** If dermoscopic, does the image look like it was captured in polarized rather than non-polarized mode?
- **Q003.** Does a circular vignette or dark corners betray a contact dermatoscope lens?
- **Q004.** Can you see signs of immersion fluid or a glass contact plate flattening the skin surface?
- **Q005.** Which device class most plausibly produced the image: smartphone, DSLR, or dedicated dermatoscope attachment?
- **Q006.** Was a flash fired, judging by hard shadows or a central bright hot spot?
- **Q007.** Could this be a grayscale en-face reflectance confocal microscopy mosaic rather than a photograph?
- **Q008.** Might this be a cross-sectional OCT or high-frequency ultrasound frame rather than a surface view?
- **Q009.** Is the image a crop or tile taken from a 3D total-body photography avatar?
- **Q010.** Is the field of view a close-up, a regional body view, or a whole-body overview?
- **Q011.** Roughly what magnification does the image represent relative to naked-eye viewing?
- **Q012.** Is the lighting diffuse and even, or directional and raking?
- **Q013.** Does the capture look multispectral or single-band (grayscale, UV, infrared) rather than standard RGB?
- **Q014.** Is Wood's-lamp or ultraviolet-induced fluorescence apparent?
- **Q015.** For a clinical photo, does the absence of surface glare suggest a cross-polarizing filter was used?
- **Q016.** Was the camera axis perpendicular to the skin, or is the lesion viewed obliquely?
- **Q017.** Does film grain, dust, or halftone dotting suggest a scanned print or slide?
- **Q018.** Does the setting look like a clinical exam room or like a patient's home snapshot?
- **Q019.** Are device overlays, timestamps, or software interface elements visible in the frame?
- **Q020.** Does the image appear to be one member of a sequential digital dermoscopy pair or series?
- **Q021.** Has the image been digitally zoomed, showing pixelation without additional optical detail?
- **Q022.** Are there telltale signs of HDR processing, beauty filters, or aggressive smartphone sharpening?
- **Q023.** Is this a composite placing clinical and dermoscopic views of the same lesion side by side?
- **Q024.** Could this be a still grabbed from video rather than a native photograph?
- **Q025.** Is the illuminant colour temperature warm (yellowish), neutral, or cool (bluish)?

## Image quality, artefacts and acquisition standards (30)

*Grounding sources: Katragadda 2017; Finnane 2017; Mirikharaji 2023; Caffery 2018; Celebi 2019*

- **Q026.** Is the lesion in sharp focus across its entire extent?
- **Q027.** Does motion blur degrade the image?
- **Q028.** Is the depth of field so shallow that parts of the lesion edge fall out of focus?
- **Q029.** Are highlights clipped (overexposed) in the lesion or surrounding skin?
- **Q030.** Is the image underexposed so that dark structures merge into shadow?
- **Q031.** How much sensor noise or grain is visible in uniform skin regions?
- **Q032.** Are JPEG block artefacts or colour banding visible?
- **Q033.** Approximately how many pixels span the lesion's longest axis?
- **Q034.** Do terminal hairs overlie and obscure part of the lesion?
- **Q035.** Are air bubbles from the immersion medium sitting over the lesion?
- **Q036.** Do glare spots or specular reflections hide surface detail?
- **Q037.** Are surgical-pen lines, drawn circles, or ink marks visible on the skin?
- **Q038.** Is a ruler or other physical size reference included in the frame?
- **Q039.** Does a colour-calibration chart or grey card appear in the image?
- **Q040.** Does a white-balance cast make normal skin look unnaturally green, blue, or magenta?
- **Q041.** Is the background uniform and non-distracting, as acquisition guidelines recommend?
- **Q042.** Is the lesion centred in the frame?
- **Q043.** Does any part of the lesion run off the image edge?
- **Q044.** Are dressings, adhesive tape, or electrodes visible?
- **Q045.** Is identifying content such as a face, tattoo, jewellery, or a text label in view?
- **Q046.** Has gel or fluid smeared to create streaking artefacts?
- **Q047.** Could pressure from the dermatoscope plate have blanched the lesion and hidden vessels?
- **Q048.** Do lens dirt, scratches, or condensation spots mar the image?
- **Q049.** Is chromatic aberration (colour fringing at high-contrast edges) present?
- **Q050.** Has the image been stretched or squashed out of its native aspect ratio?
- **Q051.** Would a dermatologist judge this image adequate for diagnostic interpretation?
- **Q052.** Have arrows, segmentation outlines, or text annotations been drawn over the image?
- **Q053.** Is there an illumination falloff gradient from the centre to the edges of the frame?
- **Q054.** Are shadows from fingers, instruments, or the dermatoscope housing present?
- **Q055.** Has colour saturation been boosted so that reds look artificially vivid?

## Anatomic site, framing and scale (25)

*Grounding sources: Katragadda 2017; Fernandez 2023; Hernández-Pérez 2024; Rotemberg 2021; Grob 1998*

- **Q056.** Which broad region is shown: head/neck, trunk, upper limb, lower limb, acral, genital, or mucosal?
- **Q057.** Does the lesion sit on chronically sun-exposed skin?
- **Q058.** Is the lesion on a palm or a sole?
- **Q059.** Does the lesion involve the nail unit?
- **Q060.** Is the lesion on the central face (the 'H-zone')?
- **Q061.** Is the lesion on the scalp, partly hidden by hair?
- **Q062.** Does the lesion lie on mucosa or the lip vermilion?
- **Q063.** Is the lesion located in an intertriginous fold?
- **Q064.** Is the ear the site of the lesion?
- **Q065.** Is the lesion on the lower leg?
- **Q066.** Are anatomic landmarks visible that would allow localisation without metadata?
- **Q067.** How many discrete lesions can be counted in the frame?
- **Q068.** Is one lesion clearly the subject, or are several candidates equally prominent?
- **Q069.** What fraction of the image area does the lesion occupy?
- **Q070.** Using any scale present, what is the estimated maximum diameter in millimetres?
- **Q071.** Does the lesion straddle a joint or flexural crease that distorts its outline?
- **Q072.** Is the lesion within or next to a scar?
- **Q073.** Is the lesion inside or adjacent to a tattoo?
- **Q074.** Is the site densely covered with pigmented hair that might mimic lesion structures?
- **Q075.** Is the lesion on the back, a site that is hard to self-examine?
- **Q076.** Are neighbouring naevi included in the photograph for comparison?
- **Q077.** Is the image orientation recoverable (e.g., is 'up' cranial)?
- **Q078.** Is the lesion on the breast or nipple–areola complex?
- **Q079.** Does the lesion involve the eyelid or periocular skin?
- **Q080.** Do body proportions or skin texture suggest a paediatric patient?

## Primary lesion morphology (ILDS terms) (30)

*Grounding sources: Nast 2016; Kittler 2016*

- **Q081.** Does the lesion appear flat and non-palpable (macule or patch)?
- **Q082.** If flat, is it under 1 cm (macule) or 1 cm or larger (patch)?
- **Q083.** Is the lesion raised as a papule (<1 cm) or as a broad plaque (≥1 cm)?
- **Q084.** Does it present as a nodule with apparent depth and a rounded profile?
- **Q085.** Is there a fluid-filled blister, and is it a vesicle or a bulla?
- **Q086.** Is the cavity filled with pus (pustule)?
- **Q087.** Could the lesion be a transient oedematous wheal?
- **Q088.** Does it look like a cyst: smooth, dome-shaped, perhaps with a punctum?
- **Q089.** Is a central punctum or dilated pore visible?
- **Q090.** Is the lesion pedunculated, hanging on a narrow stalk?
- **Q091.** Does it sit on a broad, sessile base?
- **Q092.** Is the profile dome-shaped?
- **Q093.** Is the top flat, plateau-like?
- **Q094.** Is the surface papillomatous or verrucous?
- **Q095.** Is there a central umbilication?
- **Q096.** Does the lesion grow outward (exophytic) or lie depressed below the surface?
- **Q097.** Does it resemble non-blanching purpura or petechiae?
- **Q098.** Is it a cherry-red vascular papule?
- **Q099.** Is the lesion one of a cluster (herpetiform or grouped) rather than solitary?
- **Q100.** Are multiple lesions arranged linearly, annularly, or in arcs?
- **Q101.** Does the distribution follow a dermatome or segment?
- **Q102.** Is the lesion annular with central clearing?
- **Q103.** Does a target-like set of concentric zones appear?
- **Q104.** Is the lesion polypoid?
- **Q105.** Can a burrow or serpiginous track be seen?
- **Q106.** Are thin, finger-like filiform projections present?
- **Q107.** Is the overall configuration round, oval, or irregular/geographic?
- **Q108.** Is the lesion part of a confluent eruption rather than discrete?
- **Q109.** Is a satellite lesion present near the main lesion?
- **Q110.** Does a horn-like keratin projection (cutaneous horn) arise from the lesion?

## Secondary changes and surface features (30)

*Grounding sources: Nast 2016; Rosendahl 2011; MacKie 1990; Walter 2013*

- **Q111.** Is scale present on the surface?
- **Q112.** If scaly, is the scale fine/powdery, plate-like, or thick and adherent?
- **Q113.** Is there a peripheral collarette of scale?
- **Q114.** Is haemorrhagic or serous crust present?
- **Q115.** Is the lesion ulcerated?
- **Q116.** Is there superficial erosion without full-thickness ulceration?
- **Q117.** Is fresh blood or active bleeding visible?
- **Q118.** Does the lesion ooze or carry exudate?
- **Q119.** Is there a central keratin plug or mass?
- **Q120.** Are skin markings thickened and accentuated (lichenification)?
- **Q121.** Are excoriations (scratch marks) present?
- **Q122.** Is the skin atrophic — thin, finely wrinkled, and shiny?
- **Q123.** Are open comedones visible on the surrounding skin?
- **Q124.** Is the surface fissured?
- **Q125.** Does the lesion look indurated or infiltrated?
- **Q126.** Is the lesion surrounded by an inflammatory erythematous halo?
- **Q127.** Is a depigmented white halo present around the lesion?
- **Q128.** Does the lesion have a rolled, raised border?
- **Q129.** Does the lesion look sclerotic or scar-like?
- **Q130.** Is there black necrotic eschar?
- **Q131.** Is the skin macerated (white and sodden)?
- **Q132.** Is the surface cerebriform, with brain-like ridges and fissures?
- **Q133.** Does the lesion appear 'stuck on' the skin?
- **Q134.** Is hyperkeratosis confined to the lesion margin?
- **Q135.** Does the surrounding skin appear oedematous?
- **Q136.** Are there signs of prior biopsy or treatment (sutures, curettage scar, cryotherapy blister)?
- **Q137.** Are there signs of secondary infection such as honey-coloured crust?
- **Q138.** Does the lesion look friable?
- **Q139.** Is the overlying epidermis smooth and stretched?
- **Q140.** Are follicular openings preserved, plugged, or effaced within the lesion?

## Shape, size, symmetry and border (30)

*Grounding sources: Friedman 1985; Abbasi 2004; Nachbar 1994; Henning 2007; Celebi 2019*

- **Q141.** Is the silhouette asymmetric about zero, one, or two perpendicular axes?
- **Q142.** Are colours and structures distributed asymmetrically across perpendicular axes?
- **Q143.** Is the outline smooth and regular?
- **Q144.** Is the border notched, scalloped, or ragged?
- **Q145.** Does pigment stop abruptly at the border or fade gradually into normal skin?
- **Q146.** In how many of eight peripheral segments is there an abrupt pigment cut-off (TDS border score)?
- **Q147.** Is the border blurred and poorly defined all the way around?
- **Q148.** Does the diameter exceed 6 mm?
- **Q149.** Is the diameter 7 mm or more (Glasgow minor criterion)?
- **Q150.** What is the ratio of the longest to the shortest axis?
- **Q151.** Is the outline compact and blob-like rather than lobulated or branching?
- **Q152.** How convex is the outline relative to its convex hull?
- **Q153.** Is the lesion bilobed, as if two lobes have merged?
- **Q154.** Is the long axis aligned with skin tension lines or dermatoglyphics?
- **Q155.** Is the shape nearly a perfect circle?
- **Q156.** Are islands of pigment separated from the main lesion body?
- **Q157.** Is the shape geographic, like a map?
- **Q158.** Would you call the lesion small (<3 mm), medium, or large (>15 mm)?
- **Q159.** Does a focal bulge in the outline suggest localised growth?
- **Q160.** Is the silhouette symmetric while the internal structure is asymmetric?
- **Q161.** Is the rim hypopigmented relative to the centre?
- **Q162.** Is the periphery darker than the centre?
- **Q163.** Is there a peripheral rim of tan structureless pigment?
- **Q164.** How would symmetry score on the CASH 0–2 scale?
- **Q165.** Is the border markedly more irregular on one side than the other?
- **Q166.** Are there concave indentations in the outline?
- **Q167.** Is the lesion reniform or crescent-shaped?
- **Q168.** Is the transition zone between lesion and skin narrow or wide?
- **Q169.** Does the outline contain sharp angular corners?
- **Q170.** Is the lesion more elevated on one side than the other?

## Colour (30)

*Grounding sources: Nachbar 1994; Henning 2007; Argenziano 1998; Menzies 1996; Soyer 2004*

- **Q171.** How many of the six dermoscopic colours (white, red, light brown, dark brown, blue-grey, black) are present?
- **Q172.** Is the lesion uniform in colour?
- **Q173.** Is black present anywhere?
- **Q174.** Is any blue-grey colour present?
- **Q175.** Do any white areas appear?
- **Q176.** Is red present?
- **Q177.** Is the lesion predominantly light brown or tan?
- **Q178.** Is dark brown the predominant colour?
- **Q179.** Is the lesion skin-coloured, i.e., amelanotic or hypomelanotic?
- **Q180.** Is the lesion pink without visible pigment?
- **Q181.** Is there a violaceous or purple hue?
- **Q182.** Does the lesion appear yellow or yellow-orange?
- **Q183.** Is a greenish tinge visible?
- **Q184.** Are colours arranged symmetrically or eccentrically?
- **Q185.** Is the pigment darkest centrally, peripherally, or off-centre?
- **Q186.** Does dark brown-to-black colour cover more than half of the lesion?
- **Q187.** Is the colour pattern reticular rather than homogeneous?
- **Q188.** Are multiple shades of brown present?
- **Q189.** Does the lesion look translucent?
- **Q190.** Is there a pearly-white sheen?
- **Q191.** Is redness diffuse (erythema) or confined to vessels?
- **Q192.** How strong is the colour contrast between lesion and surrounding skin?
- **Q193.** Is the lesion lighter than the surrounding skin?
- **Q194.** Is the skin around the lesion hyperpigmented?
- **Q195.** Is a grey-brown (taupe) tone present?
- **Q196.** Do colour zones blend gradually or change abruptly?
- **Q197.** Are orange structureless areas visible?
- **Q198.** Does a homogeneous blue colour cover the entire lesion?
- **Q199.** Does colour change in concentric rings?
- **Q200.** In one word, what is the lesion's dominant hue?

## Dermoscopic structures of melanocytic lesions (40)

*Grounding sources: Argenziano 1998; Menzies 1996; Kittler 2016; Soyer 2004; Henning 2007; Nachbar 1994; Argenziano 2003*

- **Q201.** Is a pigment network present?
- **Q202.** Is the network typical (thin, regular, fading peripherally) or atypical (thick, irregular, ending abruptly)?
- **Q203.** Is a negative (inverse) network visible?
- **Q204.** Are dots present, and are they regularly or irregularly distributed?
- **Q205.** Can globules (clods) be identified?
- **Q206.** Do globules form a peripheral rim, suggesting radial growth?
- **Q207.** Are radial streaks (lines) present?
- **Q208.** Are bulbous pseudopods present at the periphery?
- **Q209.** Are streaks distributed around the full circumference (starburst) or only focally?
- **Q210.** Is a blue-white veil present?
- **Q211.** Are regression structures (white scar-like areas, blue-grey peppering) seen?
- **Q212.** Are shiny white structures (crystalline/chrysalis lines) visible?
- **Q213.** Does structureless (homogeneous) area occupy more than 10% of the lesion?
- **Q214.** Which global pattern fits best: reticular, globular, homogeneous, starburst, multicomponent, or nonspecific?
- **Q215.** Does the lesion combine three or more distinct dermoscopic patterns?
- **Q216.** Does the lesion show a single colour with a symmetric pattern (both Menzies negative features)?
- **Q217.** Are multiple blue-grey dots present?
- **Q218.** Are broadened network lines present?
- **Q219.** Are angulated lines forming polygons (zig-zag pattern) visible?
- **Q220.** Is there an off-centre hyperpigmented blotch?
- **Q221.** Is there an eccentric peripheral structureless zone?
- **Q222.** How many differential structures (network, dots, globules, streaks, structureless areas) are present?
- **Q223.** How disordered is the architecture on the CASH 0–2 scale?
- **Q224.** How homogeneous are the structures (CASH homogeneity score)?
- **Q225.** Is there 'chaos', i.e., asymmetry of structure or colour?
- **Q226.** Is a cobblestone pattern of large angulated clods present?
- **Q227.** Does the network show central hyperpigmentation fading towards the edge?
- **Q228.** Are peripheral brown structureless areas present?
- **Q229.** Is a target network (network holes containing dots) present?
- **Q230.** Are the network holes roughly uniform in size?
- **Q231.** Do pigmented line segments end abruptly at the edge?
- **Q232.** Is there a blue structureless area without an overlying whitish veil?
- **Q233.** Does the lesion show a spitzoid pattern?
- **Q234.** Are tiny pigmented dots aligned along network lines?
- **Q235.** Does the lesion have a 'fried-egg' configuration with a papillated centre and flat pigmented rim?
- **Q236.** Is perifollicular hypopigmentation present within the network?
- **Q237.** Are ring-like pigmented structures seen around follicles on non-facial skin?
- **Q238.** Using the Menzies colour count, are five or more colours present?
- **Q239.** Would the 3-point checklist be positive (at least two of asymmetry, atypical network, blue-white structures)?
- **Q240.** Would the dermoscopic 7-point checklist score reach 3 or more?

## Vascular structures (25)

*Grounding sources: Rosendahl 2011; Zalaudek 2012; Menzies 2000; Kittler 2016*

- **Q241.** Are any blood vessels visible?
- **Q242.** Are sharply focused, branching arborising vessels present?
- **Q243.** Are dotted vessels present?
- **Q244.** Are glomerular (coiled) vessels present?
- **Q245.** Are hairpin (looped) vessels present?
- **Q246.** Are linear-irregular vessels present?
- **Q247.** Are comma-shaped vessels present?
- **Q248.** Are corkscrew vessels present?
- **Q249.** Are two or more vessel types present (polymorphous vessels)?
- **Q250.** Are vessels evenly distributed or randomly scattered?
- **Q251.** Do vessels crown the periphery of the lesion?
- **Q252.** Are vessels arranged like a string of pearls?
- **Q253.** Do hairpin vessels carry a whitish halo?
- **Q254.** Are milky-red areas present?
- **Q255.** Are well-demarcated red-blue lacunae present?
- **Q256.** Is a red pseudonetwork around follicles (strawberry pattern) present?
- **Q257.** Do vessels form a reticular, net-like pattern?
- **Q258.** Are dark red-black thrombosed lacunae present?
- **Q259.** Are vessels concentrated centrally?
- **Q260.** Are vessels crisp (superficial) or blurred (deeper)?
- **Q261.** How thick is the widest vessel relative to lesion size?
- **Q262.** Are vessels dilated and tortuous at the lesion edge?
- **Q263.** Do vessels radiate from the centre?
- **Q264.** Does diffuse erythema obscure vessel morphology?
- **Q265.** Is vessel visibility likely limited by high background melanin?

## Non-melanocytic lesion criteria (BCC, AK/Bowen/SCC, seborrhoeic keratosis, others) (35)

*Grounding sources: Menzies 2000; Zalaudek 2012; Kandolf 2024; Stratigos 2026; Rosendahl 2011; Olsen 1991*

- **Q266.** Is a pigment network absent, as required by the Menzies BCC criteria?
- **Q267.** Are leaf-like areas present?
- **Q268.** Are spoke-wheel areas present?
- **Q269.** Are large blue-grey ovoid nests present?
- **Q270.** Are multiple non-aggregated blue-grey globules present?
- **Q271.** Are shiny white-red structureless areas present?
- **Q272.** Are there multiple small erosions or a small ulceration?
- **Q273.** Are short fine superficial telangiectasias present at the margin?
- **Q274.** Are concentric structures present?
- **Q275.** Are milia-like cysts visible?
- **Q276.** Are comedo-like openings present?
- **Q277.** Is there a light-brown structureless centre with a sharp edge, as in a flat seborrhoeic keratosis or lentigo?
- **Q278.** Is the border sharply demarcated and moth-eaten?
- **Q279.** Are fingerprint-like structures present?
- **Q280.** Is there background erythema with yellow-white follicular scale?
- **Q281.** What Olsen keratotic grade fits (I: barely visible, II: moderate, III: thick hyperkeratotic)?
- **Q282.** Are white circles around follicles present?
- **Q283.** Are clustered glomerular vessels with surface scale present?
- **Q284.** Is there a central keratin mass surrounded by white structureless zones?
- **Q285.** Are blood spots embedded in keratin?
- **Q286.** Are rosettes (four white points in a square) visible?
- **Q287.** Is there a central white scar-like patch with a delicate peripheral network?
- **Q288.** Is there a white collarette around a red vascular blush?
- **Q289.** Is a yellowish lobular structure with crown vessels present?
- **Q290.** Does a soft, skin-coloured lesion with a wrinkled surface suggest a dermal naevus or neurofibroma?
- **Q291.** Are brown dots or globules arranged in lines within a scaly patch?
- **Q292.** Does a scaly plaque with evenly distributed dotted vessels suggest an inflammatory dermatosis rather than a tumour?
- **Q293.** Does the lesion fit along the AK–Bowen–SCC continuum of increasing thickness and ulceration?
- **Q294.** Are scattered brown dots on a pink background present?
- **Q295.** Is there a peripheral erythematous rim around a keratotic centre?
- **Q296.** Is the lesion a pink, scaly, well-demarcated plaque?
- **Q297.** Is the lesion consistent with a vascular tumour rather than a keratinocytic one?
- **Q298.** Are black areas sitting within blue lacunae, raising thrombosed angioma versus nodular melanoma?
- **Q299.** Are thick curved 'fat fingers' present, as seen in seborrhoeic keratosis?
- **Q300.** Which category fits best: melanocytic, keratinocytic, vascular, adnexal, fibrous, or inflammatory?

## Site-specific patterns (acral, facial, nail, mucosal) (25)

*Grounding sources: Forsea 2026; Fernandez 2023; Enechukwu 2023; Kittler 2016*

- **Q301.** On acral skin, does pigment follow the furrows (parallel furrow pattern)?
- **Q302.** Does pigment lie on the ridges (parallel ridge pattern)?
- **Q303.** Is a lattice-like pattern present?
- **Q304.** Is a fibrillar pattern present?
- **Q305.** Is there diffuse irregular pigmentation on acral skin?
- **Q306.** On facial skin, do follicular openings create a pseudonetwork?
- **Q307.** Are pigmented follicular openings asymmetric?
- **Q308.** Is an annular-granular pattern present around follicles?
- **Q309.** Are rhomboidal structures present?
- **Q310.** Are follicular openings obliterated by pigment?
- **Q311.** Is a grey 'circle within a circle' pattern visible?
- **Q312.** Is there a longitudinal melanonychia band in a nail?
- **Q313.** Is the band 3 mm or wider, or more than one-third of the nail width?
- **Q314.** Are the lines in the band regular in colour, spacing, and thickness?
- **Q315.** Is Hutchinson's sign (periungual pigment) present?
- **Q316.** Is the nail plate dystrophic or destroyed?
- **Q317.** Are red-purple globules with proximal clearing suggestive of subungual haemorrhage?
- **Q318.** On mucosa, are structureless grey or blue areas present?
- **Q319.** On the scalp, is pigment interrupted by hair follicles?
- **Q320.** On the heel or sole, do blood spots suggest traumatic haemorrhage?
- **Q321.** On the lip, is there a well-circumscribed brown macule?
- **Q322.** Are dermatoglyphic lines visible running through the lesion?
- **Q323.** Is there granulation-like tissue at the nail fold?
- **Q324.** On the face, is the lesion a light-brown macule with fine reticulation?
- **Q325.** On genital skin, does the lesion show ring-like or parallel line patterns?

## Triage, gestalt and evolution (30)

*Grounding sources: Grob 1998; Gaudy-Marqueste 2017; Rogers 2016; Rosendahl 2011; Bourne 2012; Kittler 2006; MacKie 1990; Walter 2013; Argenziano 2003*

- **Q326.** At a glance, does the lesion look clearly benign (TADA step 1)?
- **Q327.** Is the lesion the 'ugly duckling' among the patient's naevi?
- **Q328.** Is it the only pigmented lesion with a different colour from the rest?
- **Q329.** Would two independent experts likely agree on the diagnosis for this image?
- **Q330.** Is the lesion pigmented, non-pigmented, or partly pigmented, determining which algorithm applies?
- **Q331.** Would a non-expert using this image refer the patient?
- **Q332.** Is suspicion driven mainly by colour or mainly by structure?
- **Q333.** Compared with a prior image, if available, is there evidence of change?
- **Q334.** Has the lesion grown asymmetrically relative to baseline?
- **Q335.** Have new colours emerged since baseline?
- **Q336.** Have new structures emerged since baseline?
- **Q337.** Is the lesion new, absent from the baseline image?
- **Q338.** Does the lesion lack specific features yet still raise concern?
- **Q339.** Is this a pink, raised lesion requiring separate triage?
- **Q340.** Does the lesion fit the EFG rule (elevated, firm, growing)?
- **Q341.** Could this be a nodular melanoma that would fail ABCD criteria?
- **Q342.** Would a primary-care clinician and a dermatologist triage this image differently?
- **Q343.** Does the lesion share a 'signature' pattern with the patient's other naevi?
- **Q344.** How confident would you be in a benign versus malignant call: low, moderate, or high?
- **Q345.** Would the weighted Glasgow 7-point checklist total reach 4?
- **Q346.** On a 5-point suspicion scale, would you rate the lesion 4 or 5?
- **Q347.** Should site-specific (acral, facial, nail) rather than trunk algorithms be applied?
- **Q348.** Is dermoscopy needed to judge this lesion, or is the clinical image sufficient?
- **Q349.** Would the lesion be considered equivocal enough to warrant adjunct imaging such as RCM?
- **Q350.** Might this be a collision lesion with two different components?
- **Q351.** Is the lesion one of many similar seborrhoeic keratoses (an eruptive pattern)?
- **Q352.** Is the lesion solitary on otherwise unblemished skin, making intra-patient comparison impossible?
- **Q353.** Are halo naevi or regressing lesions visible elsewhere in the frame?
- **Q354.** Would 'chaos and clues' be positive (chaos plus at least one clue)?
- **Q355.** Could BLINCK's patient-concern or change factors be inferred from the image at all?

## Skin tone and background skin (30)

*Grounding sources: Ware 2020; Weir 2025; Lipnick 2026; Pryor 2026; Chardon 1991; Groh 2021; Daneshjou 2022a; Wen 2022*

- **Q356.** Where does the background skin fall on the 10-point Monk Skin Tone scale?
- **Q357.** What Fitzpatrick type might be guessed, bearing in mind it encodes sunburn propensity rather than colour?
- **Q358.** Would the ITA° of normal skin be very light, light, intermediate, tan, brown, or dark?
- **Q359.** Is the surrounding skin evenly pigmented or mottled?
- **Q360.** Is erythema visible on surrounding skin, or could melanin be masking it?
- **Q361.** Does inflammation appear violaceous, grey, or dark brown rather than red?
- **Q362.** Is post-inflammatory hyperpigmentation present nearby?
- **Q363.** Is post-inflammatory hypopigmentation present nearby?
- **Q364.** Are ephelides (freckles) visible on surrounding skin?
- **Q365.** Are solar lentigines visible on surrounding skin?
- **Q366.** Does the surrounding skin show chronic sun damage such as wrinkling and mottled pigmentation?
- **Q367.** Is poikiloderma present?
- **Q368.** Can veins be seen through the skin, suggesting thin, fair skin?
- **Q369.** Is a tan line visible?
- **Q370.** Are there vitiligo-like depigmented areas?
- **Q371.** Is a pigmentary demarcation (Futcher) line visible?
- **Q372.** Does the background skin look oily or shiny?
- **Q373.** Is the background skin dry and flaking?
- **Q374.** Is the background densely hair-bearing?
- **Q375.** Is lesion–skin contrast reduced because lesion and skin pigment are similar?
- **Q376.** Might high baseline melanin make the pigment network look more prominent than in fair skin?
- **Q377.** Does the white balance make skin-tone estimation unreliable?
- **Q378.** Would the same skin map to a different Monk category under another illuminant?
- **Q379.** Are small porcelain-white macules (guttate hypomelanosis) scattered on surrounding skin?
- **Q380.** Are dermatosis papulosa nigra–like papules present nearby?
- **Q381.** Is the volar skin visibly lighter than the dorsal skin?
- **Q382.** Are blue-grey background areas of dermal melanocytosis unrelated to the lesion present?
- **Q383.** Is enough non-lesional skin visible to estimate skin tone reliably?
- **Q384.** Is there rosacea-like baseline redness on surrounding skin?
- **Q385.** Does the skin tone belong to a group underrepresented in public datasets (Monk 7–10 / Fitzpatrick V–VI)?

## Chromophores and optical properties (30)

*Grounding sources: Tsumura 1999; Tsumura 2003; Claridge 2003; Moncrieff 2002; Donner 2008; Jensen 2001; Kuzmina 2022; Glud 2009*

- **Q386.** Does the pigment look brown (epidermal melanin) or blue-grey (dermal melanin, Tyndall effect)?
- **Q387.** Would a melanin-concentration map show a focal maximum?
- **Q388.** Would a haemoglobin map show increased blood within the lesion?
- **Q389.** Does the lesion centre show blood displacement (reduced haemoglobin)?
- **Q390.** Is there a whitish zone suggesting altered dermal collagen scattering?
- **Q391.** Does the lesion scatter light more (look milkier) than surrounding skin?
- **Q392.** In optical-density space, does the lesion separate along the melanin axis, the haemoglobin axis, or both?
- **Q393.** Would comparing short- and long-wavelength bands help separate melanin from haemoglobin here?
- **Q394.** Does the lesion show raised melanin but normal blood, i.e., pigmented without inflammation?
- **Q395.** Does subsurface scattering soften the lesion edges?
- **Q396.** Is the colour explainable by a two-chromophore (melanin + haemoglobin) model, or do other absorbers seem involved?
- **Q397.** Do pigment structures appear weaker in the red channel than in the blue channel?
- **Q398.** Is the green channel the highest-contrast channel for vessels?
- **Q399.** Is the lesion visible in all three RGB channels?
- **Q400.** Does the blood look oxygenated (bright red) or deoxygenated (dark purple-blue)?
- **Q401.** Is there rust-brown colouring suggesting haemosiderin deposition?
- **Q402.** Is exogenous pigment (tattoo ink, graphite, amalgam) likely?
- **Q403.** Is the lesion translucent enough for underlying vessels to show through?
- **Q404.** Does pigment seem to sit within the stratum corneum (crisp, dark, superficial) rather than deeper?
- **Q405.** Judging by colour, does the lesion look optically superficial or deep?
- **Q406.** Does the lesion absorb strongly across all wavelengths, appearing near-black and neutral?
- **Q407.** Is chromophore concentration spatially heterogeneous (mottled) or homogeneous?
- **Q408.** Could illumination assumptions plausibly bias a melanin estimate in this image?
- **Q409.** Would scale or keratin create a whitish layer that violates linear chromophore-separation assumptions?
- **Q410.** Do ulceration or crust make chromophore inversion unreliable for this lesion?
- **Q411.** Is there a bright superficial reflection from a thin epidermis?
- **Q412.** Is there a depth gradient, e.g., a brown centre with a bluish periphery?
- **Q413.** Is hue saturation high, consistent with concentrated melanin?
- **Q414.** Would appearance change substantially between polarized and non-polarized illumination?
- **Q415.** Do apparent chromophore levels fall outside physiological ranges (implausible colour)?

## Surface microrelief, texture and reflectance (25)

*Grounding sources: Ghosh 2008; Sohaib 2013; Graham 2013; Linming 2018; Weyrich 2006*

- **Q416.** Is the fine furrow network of skin microrelief visible across the lesion?
- **Q417.** Does the lesion efface the normal skin furrows?
- **Q418.** Do the furrows deviate around the lesion?
- **Q419.** Are specular highlights present, indicating a moist or shiny surface?
- **Q420.** Are highlights broad (smooth surface) or fragmented (rough surface)?
- **Q421.** Would the surface likely feel rough to the touch?
- **Q422.** Are the pores around the lesion visible and enlarged?
- **Q423.** Do wrinkles cross the lesion?
- **Q424.** Does raking light reveal elevation through cast shadows?
- **Q425.** Could height (topography) be inferred from shading cues?
- **Q426.** Does the surface look wet from exudate or applied ointment?
- **Q427.** Is surface texture uniform or heterogeneous?
- **Q428.** Is the surface pitted with small depressions?
- **Q429.** Is the surface shiny and taut over a nodule?
- **Q430.** Does the dermatoscope's light ring reflect as concentric rings on the surface?
- **Q431.** Does glare differ between lesion and surrounding skin?
- **Q432.** Is there a granular texture?
- **Q433.** Is the surface cobbled or lobulated?
- **Q434.** Do furrows form a cross-hatched pattern?
- **Q435.** Does the surface look velvety?
- **Q436.** Is the lesion flat enough for uniform contact dermoscopy?
- **Q437.** Does the texture have a preferred orientation (anisotropic grain)?
- **Q438.** Is there peau d'orange dimpling?
- **Q439.** Is fine vellus hair visible on the lesion?
- **Q440.** Does texture differ between the lesion centre and its periphery?

## Computational image features (25)

*Grounding sources: Celebi 2019; Codella 2018; Mirikharaji 2023; Kim 2024a (MONET); Nachbar 1994*

- **Q441.** Could the lesion be segmented automatically as a single clean closed boundary?
- **Q442.** Is lesion–background contrast high enough for threshold-based segmentation?
- **Q443.** Would hair-removal preprocessing be needed before segmentation?
- **Q444.** Is the boundary ambiguous enough that annotators would disagree?
- **Q445.** Is GLCM contrast (texture heterogeneity) within the lesion high?
- **Q446.** Is texture entropy within the lesion high?
- **Q447.** Is the lesion texture homogeneous (high GLCM homogeneity/energy)?
- **Q448.** How large is the colour variance inside the lesion mask?
- **Q449.** Is mean lesion colour distinct from mean skin colour in a perceptual space (large ΔE)?
- **Q450.** Does the colour histogram show multiple modes?
- **Q451.** Is there a radial intensity gradient from centre to edge?
- **Q452.** Is the intensity gradient across the boundary steep or shallow?
- **Q453.** How high is the fractal dimension or irregularity index of the border?
- **Q454.** Is the geometric centroid offset from the pigment-weighted centre of mass?
- **Q455.** Does the image contain spurious cues (rulers, markers) a CNN could exploit as shortcuts?
- **Q456.** Would concept-scoring models flag concepts such as ulceration or erythema in this image?
- **Q457.** Would saturated-pixel regions corrupt colour features?
- **Q458.** Is colour-constancy normalisation needed before feature extraction?
- **Q459.** Would rotating the image change its interpretation?
- **Q460.** Would key structures survive downsampling to a standard CNN input size such as 224 px?
- **Q461.** Is periodic texture (network) detectable through frequency analysis?
- **Q462.** Would a local texture descriptor (e.g., LBP) distinguish the lesion centre from its rim?
- **Q463.** Are there enough background-skin pixels to compute normalisation statistics?
- **Q464.** Does the image require multi-lesion rather than single-lesion segmentation?
- **Q465.** Would a saliency map likely concentrate on the lesion rather than on artefacts?

## Dataset provenance, synthetic data and ML suitability (25)

*Grounding sources: Tschandl 2018; Pacheco 2020; Groh 2021; Hernández-Pérez 2024; Combalia 2022; Bissoto 2021; Sinha 2024; Kim 2024b; Rezk 2022; Daneshjou 2022b*

- **Q466.** Does the image match the visual style of ISIC/HAM10000 dermoscopic images?
- **Q467.** Does it resemble smartphone clinical images such as those in PAD-UFES-20?
- **Q468.** Is the lesion type likely absent from typical training data (out-of-distribution)?
- **Q469.** Does the image show generative-model artefacts such as repeated texture or implausible hair?
- **Q470.** Are there blending seams typical of physics-based or 3D-rendered synthetic lesions?
- **Q471.** Does the lesion look pasted onto skin with mismatched lighting?
- **Q472.** Are watermarks or textbook/atlas styling present?
- **Q473.** Does the image look web-scraped from a clinical atlas, as in Fitzpatrick17k?
- **Q474.** Is the image an educational figure with captions or panel letters?
- **Q475.** Would the label for this image more likely be histopathology-confirmed or clinical consensus?
- **Q476.** Would per-feature labels for this image be unreliable because the features are ambiguous?
- **Q477.** Does the image depict a rare diagnostic class?
- **Q478.** Would the image serve as a skin-tone fairness benchmark case?
- **Q479.** Does the image look like it came from a biopsy-selected clinic population?
- **Q480.** Has the image been heavily augmented (rotation, colour jitter, flipping)?
- **Q481.** Is the image a near-duplicate of another view of the same lesion?
- **Q482.** Does the crop omit context needed for patient-level analysis?
- **Q483.** Are demographic cues visible that could confound a classifier?
- **Q484.** Does the image appear style-transferred, e.g., with artificially darkened skin?
- **Q485.** Has the image been anonymised by blurring?
- **Q486.** Does lesion texture resolution mismatch skin texture resolution, suggesting a generated region?
- **Q487.** Does the image appear upscaled by super-resolution?
- **Q488.** Is this a hard case comparable to BCN20000's difficult anatomical sites?
- **Q489.** Would this image belong to a non-lesion or 'other' category?
- **Q490.** Is the image cropped to the fixed square size typical of challenge datasets?

## 3D, total-body and longitudinal imaging (10)

*Grounding sources: Cerminara 2023; Kittler 2006; Salerni 2013; Soyer 2025; Primiero 2024*

- **Q491.** Is the image a textured 3D total-body mesh with visible texture seams?
- **Q492.** Is the lesion mapped to a coordinate on a body avatar?
- **Q493.** Could the lesion be matched across visits using surrounding landmarks?
- **Q494.** For a dated pair, is the follow-up interval short-term (~3 months) or long-term (≥6 months)?
- **Q495.** Could apparent change between images be explained by angle or lighting rather than true growth?
- **Q496.** Does a 3D view reveal the lesion's elevation profile?
- **Q497.** Is the naevus count in the visible region high?
- **Q498.** Is the texture resolution of a total-body crop adequate for dermoscopic-level features?
- **Q499.** Does body-surface curvature distort the apparent lesion shape?
- **Q500.** Is the lesion visible in multiple overlapping camera views?
