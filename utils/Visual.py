from sklearn import metrics

def claification_report(label, pred, name):
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == "KSC":
        target_names = ['Scrub', 'Willow swamp', 'Cabbage palm hammock', 'Cabbage palm/oak hammock', 'Slash pine',
                        'Oak/broadleaf hammock',
                        'Hardwood swamp', 'Graminoid marsh', 'Spartine marsh', 'Cattail marsh', 'Salt marsh',
                        'Mud flats', 'Water']
    elif name == 'SA':
        target_names = ['Brocoli-green-weeds-1', 'Brocoli-green-weeds-2', 'Fallow', 'Fallow-rough-plow', 'Fallow-smooth', 'Stubble', 'Celery',
                        'Grapes-untrained', 'Soil-vinyard-develop', 'Corn-senesced-green-weeds', 'Lettuce-romaine-4wk', 'Lettuce-romaine-5wk', 'Lettuce-romaine-6wk', 'Lettuce-romaine-7wk',
                        'Vinyard-untrained','Vinyard-vertical-trellis']
    elif name == 'UP':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    elif name == 'HU_tif':
        target_names = ['Grass_healthy', 'Grass_stressed', 'Grass_synthetic', 'Tree', 'Soil', 'Water', 'Residential',
                        'Commercial', 'Road', 'Highway', 'Railway', 'Parking_lot1', 'Parking_lot2', 'Tennis_court',
                        'Running_track']
    elif name == 'BOT':
        target_names = ['Water', 'Hippograss', 'Floodplaingrasses1', 'Floodplaingrasses2',
                        'Reeds1', 'Riparian', 'Fierscar2',
                        'Island interior', 'Acacia woodlands', 'Acacia shrublands', 'Acacia grasslands',
                        'Shortmopane', 'Mixedmopane', 'Exposedsoils']
    elif name == 'CH':
        target_names = ['Rice stubble', 'Grassland', 'Elm', 'Ash tree','Pagoda Tree', 'Vegetable field', 'Poplar',
                        'Soybean', 'Black locust', 'Rice', 'Water','Willow', 'Acer negundo', 'Goldenrain tree',
                        'Peach tree', 'Corn', 'Pear tree','Lotus leaf','Building']

    classification_report = metrics.classification_report(label, pred, target_names=target_names)
    return classification_report