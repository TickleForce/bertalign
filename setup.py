import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='grc-bertalign',
    version='0.0.1',
    author='TickleForce',
    author_email='koderkk@gmail.com',
    description='Translation alignment with sentence transformers.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/TickleForce/grc-bertalign',
    project_urls = {
        "Bug Tracker": "https://github.com/TickleForce/grc-bertalign/issues"
    },
    license='MIT',
    packages=['bertalign'],
    install_requires=['numba', 'faiss-cpu'],
)
