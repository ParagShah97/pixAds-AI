class_names = ['airport_terminal',
                        'auditorium',
                        'bedroom',
                        'bookstore',
                        'bus_station-indoor',
                        'clothing_store',
                        'computer_room',
                        'food_court',
                        'jewelry_shop',
                        'railroad_track']

class_actual_name = ['Airport Terminal',
                     'Auditorium',
                     'Bedroom',
                     'Book Store',
                     'Bus Station and Buses',
                     'Cloths Store',
                     'Computer and Gadgets Room',
                     'Food Court',
                     'Jewelry Shop',
                     'Railways']
class_description =  [
    "A terminal where passengers embark and disembark from aircraft.",
    "A large room or hall used for public gatherings, typically lectures or performances.",
    "A room used for sleeping, typically containing a bed and other furniture.",
    "A store where books are sold.",
    "An area where buses arrive and depart.",
    "A store selling clothes and accessories.",
    "A room equipped with computers for work or study.",
    "An area in a shopping center or airport where various food vendors sell food.",
    "A shop specializing in jewelry.",
    "A track or train or a train stations with passengers."
]

data = {
    'Class Name': class_actual_name,
    'Class Key': class_names,
    'Description': class_description,
    '# Samples': [5000] * len(class_names)
}

about_dataset = '''
Explore the MIT Places dataset, a comprehensive collection designed for advancing scene recognition and classification 
in computer vision. This dataset features a diverse array of images encompassing of around 365 classes from various places
we sampled the dataset with 10 classes for our usecase. Each image is meticulously labeled with its corresponding
scene category, making it ideal for training and evaluating algorithms in scene understanding, object detection,
and semantic segmentation. Researchers and developers utilize the MIT Places dataset to push the boundaries of
visual recognition systems, harnessing its breadth and depth to innovate in fields requiring robust scene
analysis and classification capabilities.
'''

introduction_to_application = '''Introducing our innovative application that leverages image analysis to deliver highly relevant ads and personalized messages based on the pictures you upload. By examining your photos, our app tailors advertisements and communications to match your interests and preferences, ensuring a unique and engaging user experience. Enjoy a new level of personalization and relevance with every interaction.'''

predicted_label_messages = {
    'airport_terminal': 'Hope you have a wonderful flight ahead, dont forgot to pack you bags and stuff your food.',
    'auditorium': 'Hope you had a wonderful auditorium experience.',
    'bedroom': 'A good night sleep in a quite bedroom, improves health.',
    'bookstore': 'I hope you pick the best book, which worth your time.',
    'bus_station-indoor': 'Road journeys are always been in good memories.',
    'clothing_store': 'Bright cloths suits the best.',
    'computer_room': 'The perfect gadget to have!',
    'food_court': 'Best food is the food served Hot!!',
    'jewelry_shop': 'Its expensive, though the best jewelry to have!',
    'railroad_track': 'Train are the most memorable journey of all time.'
    
}

urls_dict = {
    'airport_terminal' : {
        'videos': ['https://www.youtube.com/watch?v=htO1oFwCzik',
                   'https://www.youtube.com/watch?v=uQHhYRuaEtM',
                   'https://www.youtube.com/watch?v=WdSkkN2OKvs'],
        'images' : ['ads_imgs/travel_1.jpg', 'ads_imgs/dominos_2.jpg']
    },
    'auditorium' : {
        'videos': ['https://www.youtube.com/watch?v=KiEeIxZJ9x0',
                   'https://www.youtube.com/watch?v=ZAYhcbfHKPc'],
        'images' : ['ads_imgs/popcorn_1.jpeg',
                    'ads_imgs/popcorn_2.jpeg',
                    'ads_imgs/dominos_2.jpg']
    },
    'bedroom' : {
        'videos': ['https://youtu.be/Y2AnauPX1hM?si=RJ7P4OKtzvqiDdgw',
                   'https://youtu.be/wtCP_zNDaXU?si=ItVHE82VA2NOIwTy',
                   'https://www.youtube.com/watch?v=dM5H0yoCEOc',
                   'https://youtu.be/ecsVyIOY-Ig?si=YSSAA43IxBbSLvCE',
                   'https://youtu.be/7qdt0X4fEbk?si=W7we0U8ZwJAdkUnG',
                   'https://youtu.be/kjVf8ptmDmk?si=G_6q-hn4GV7len2d'],
        'images' : ['ads_imgs/bedroom_1.jpeg',
                    'ads_imgs/bedroom_2.jpeg']
    },
    'bookstore' : {
        'videos': ['https://youtu.be/XO5phda0gXM?si=L2EZ7lb6m97qsCRa',
                   'https://youtu.be/RSffykkToW8?si=98y6HhD9TBTKl79g',
                   'https://youtu.be/lwdNcMkOiTM?si=b9vTDq1a9IbJIJd6',
                   'https://youtu.be/cdoPHJ2AGIE?si=qeczZ2-hbXW2yE4o'],
        'images' : ['ads_imgs/bookstore_1.jpeg',
                    'ads_imgs/bookstore_2.jpeg']
    },
    'bus_station-indoor' : {
        'videos': ['https://youtu.be/50I-fGKcZ58?si=YKTODDIJjWdxvlh-',
                   'https://youtu.be/79phlhutGLg?si=wYAIYV9SmEN1SMgj',
                   'https://www.youtube.com/watch?v=qBMpVjywkLQ',
                   'https://www.youtube.com/watch?v=NuWS4XGb-G4'],
        'images' : ['ads_imgs/subway_1.jpg',
                    'ads_imgs/travel_2.jpg']
    },
    'clothing_store' : {
        'videos': ['https://youtu.be/zY3atcmLBag?si=3sFZhQ2bLokcByw4',
                   'https://youtu.be/Ob570Cescv4?si=kGlt_gjlQHjkOYFe',
                   'https://www.youtube.com/watch?v=jxXTyO16Fe0',
                   'https://www.youtube.com/watch?v=Dt4Zs0lhnCM',
                   'https://youtu.be/DdFnz8j9trQ?si=2CpJNGKZa1PpCZw6',
                   'https://youtu.be/Xe3evWq5tas?si=oxJ90F1THN42Z9XY'],
        'images' : ['ads_imgs/cloth_1.jpeg',
                    'ads_imgs/cloth_2.png']
    },
    'computer_room' : {
        'videos': ['https://youtu.be/3S5BLs51yDQ?si=RH-DNK-EYq4EUfzt',
                   'https://youtu.be/oxMBpN0WX5A?si=Lv2yOpBwXjEJhCtc'],
        'images' : ['ads_imgs/computer_1.jpg',
                    'ads_imgs/computer_2.jpg']
    },
    'food_court' : {
        'videos': ['https://youtu.be/79phlhutGLg?si=wYAIYV9SmEN1SMgj',
                   'https://www.youtube.com/watch?v=6pX8MfYkebk',
                   'https://youtu.be/hBkk0bbn2c0?si=I_xrulNrQ8xgQIFY'],
        'images' : ['ads_imgs/subway_1.jpg',
                    'ads_imgs/mc_d_1.jpg',
                    'ads_imgs/dominos_1.jpg']
    },
    'jewelry_shop' : {
        'videos': ['https://youtu.be/sNiKPw4lagY?si=UR69jrlgWJDE4LjT',
                   'https://www.youtube.com/watch?v=sNiKPw4lagY',
                   'https://www.youtube.com/watch?v=KQ__0x-9KTs',
                   'https://www.youtube.com/watch?v=k5A4zWJ2czM'],
        'images' : ['ads_imgs/jew_1.jpeg',
                    'ads_imgs/tanishq_1.jpg']
    },
    'railroad_track' : {
        'videos': ['https://youtu.be/V3-EIFeM27U?si=lGr4-IBG-7f8_miJ',
                   'https://youtu.be/hnukpTmSSKA?si=mkzsBQgmL64yWckL',
                   'https://youtu.be/_c-PFPmVGnc?si=yE2VkZd-j8xX9byX'],
        'images' : ['ads_imgs/travel_1.jpg',
                    'ads_imgs/travel_2.jpg']
    },
}