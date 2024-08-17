from setuptools import setup, find_packages

setup(
    name='car_classification',
    version='0.1',
    description='Car Make, Model, and Year Classification',
    author='Gaurav Upadhyay',
    author_email='gaurav@gaurav.com',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'opencv-python',
        'matplotlib'
    ],
    entry_points={
        'console_scripts': [
            'train-car-classification=car_classification.train:main',
            'evaluate-car-classification=car_classification.evaluate:main',
            'predict-car-classification=car_classification.predict:main'
        ]
    }
)
