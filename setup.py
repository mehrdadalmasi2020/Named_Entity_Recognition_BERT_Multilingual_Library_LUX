from setuptools import setup, find_packages
import os

# Disable TensorFlow to prevent circular import issues with transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Disable TensorFlow

setup(
    name='Named_Entity_Recognition_BERT_Multilingual_Library_LUX',
    version='0.1.14',
    description=(
        'A comprehensive multilingual Named Entity Recognition (NER) library leveraging BERT. '
        'Supports key information extraction tasks across various domains such as biomedical, environmental, and technological.'
    ),
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author="Mehrdad ALMASI, Demival VASQUES FILHO, Tugce KARATAS",
    author_email="mehrdad.al.2023@gmail.com, demival.vasques@uni.lu, tugce.karatas@uni.lu",
    url='https://github.com/mehrdadalmasi2020/Named_Entity_Recognition_BERT_Multilingual_Library_LUX',
    project_urls={
        "Documentation": "https://github.com/mehrdadalmasi2020/Named_Entity_Recognition_BERT_Multilingual_Library_LUX/wiki",
        "Source": "https://github.com/mehrdadalmasi2020/Named_Entity_Recognition_BERT_Multilingual_Library_LUX",
        "Tracker": "https://github.com/mehrdadalmasi2020/Named_Entity_Recognition_BERT_Multilingual_Library_LUX/issues",
    },
    packages=find_packages(include=['Named_Entity_Recognition_BERT_Multilingual_Library_LUX', 'Named_Entity_Recognition_BERT_Multilingual_Library_LUX.*']),
    install_requires=[
        'torch>=1.8.0',            # PyTorch as the main backend
        'transformers>=4.10.0',    
        'datasets>=2.1.0',
        'evaluate>=0.2.2',
        'numpy>=1.19.5',
        'seqeval>=1.2.2',
    ],
    extras_require={
        "dev": ["pytest>=6.2", "black", "flake8", "isort"],
        "docs": ["mkdocs", "mkdocs-material"],
    },
    python_requires='>=3.7',
    keywords='Named Entity Recognition, BERT, Multilingual NLP, Information Extraction, NER, Key Information Extraction',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing :: Linguistic',
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
    ],
    entry_points={
        'console_scripts': [
            'ner-multilingual=Named_Entity_Recognition_BERT_Multilingual_Library_LUX.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
