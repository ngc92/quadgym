from setuptools import setup, find_packages

setup(name='gym_quadrotor',
      version='0.1.2',
      install_requires=['gym', 'numpy'],
      test_requires=["pytest", "mock"],
      packages=find_packages()
      )
