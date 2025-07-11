from setuptools import setup, find_packages

setup(
    name="lora_image_preprocessor",
    version="0.1.0",
    packages=find_packages(),
    author="Babka Yoshka",
    author_email="b.yoshka@yandex.com",
    description="A tool to preprocess images for LoRA training.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/lora_image_preprocessor",  # Replace with your URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
