
# ignore all "UNUSED" comments at the end of some lines. Theyre leftovers I forgot to delete

vars_to_plot = [
#'LANDN -- Land-sea coverage (nearest neighbor) [0=sea; 1=land] -- 0[-] SFC="Ground or water surface"',
#'ICEC -- Ice cover [Proportion] -- 0[-] SFC="Ground or water surface"',
#'SUNSD -- SunShine duration [s] -- 0[-] SFC="Ground or water surface"',
]
data_vars = [
# https://www.nco.ncep.noaa.gov/pmb/docs/on388/table2.html
'LANDN -- Land-sea coverage (nearest neighbor) [0=sea; 1=land] -- 0[-] SFC="Ground or water surface"',
'ICEC -- Ice cover [Proportion] -- 0[-] SFC="Ground or water surface"',
'SUNSD -- SunShine duration [s] -- 0[-] SFC="Ground or water surface"',
'RH -- Relative humidity [%] -- 2[m] HTGL="Specified height level above ground"',  # UNUSED
# seems wonky
#'SNOD -- Snow depth [m] -- 0[-] SFC="Ground or water surface"',  # UNUSED
'UGRD -- u-component of wind [m/s] -- 10[m] HTGL="Specified height level above ground"',  # UNUSED
'VGRD -- v-component of wind [m/s] -- 10[m] HTGL="Specified height level above ground"',  # UNUSED
'PRES -- Pressure [Pa] -- 0[-] SFC="Ground or water surface"',  # UNUSED
'HGT -- Geopotential height [gpm] -- 0[-] SFC="Ground or water surface"',  # UNUSED
'TMP -- Temperature [C] -- 0[-] SFC="Ground or water surface"',  # UNUSED

'HGT -- Geopotential height [gpm] -- 90000[Pa] ISBL="Isobaric surface"',  # UNUSED
'TMP -- Temperature [C] -- 90000[Pa] ISBL="Isobaric surface"',  # UNUSED
'RH -- Relative humidity [%] -- 90000[Pa] ISBL="Isobaric surface"',  # UNUSED
'VVEL -- Vertical velocity (pressure) [Pa/s] -- 90000[Pa] ISBL="Isobaric surface"',  # UNUSED
'UGRD -- u-component of wind [m/s] -- 90000[Pa] ISBL="Isobaric surface"',  # UNUSED
'VGRD -- v-component of wind [m/s] -- 90000[Pa] ISBL="Isobaric surface"',  # UNUSED
'ABSV -- Absolute vorticity [1/s] -- 90000[Pa] ISBL="Isobaric surface"',  # UNUSED
'CLWMR -- Cloud mixing ratio [kg/kg] -- 90000[Pa] ISBL="Isobaric surface"',  # UNUSED

'HGT -- Geopotential height [gpm] -- 70000[Pa] ISBL="Isobaric surface"',  # UNUSED
'TMP -- Temperature [C] -- 70000[Pa] ISBL="Isobaric surface"',  # UNUSED
'RH -- Relative humidity [%] -- 70000[Pa] ISBL="Isobaric surface"',  # UNUSED
'VVEL -- Vertical velocity (pressure) [Pa/s] -- 70000[Pa] ISBL="Isobaric surface"',  # UNUSED
'UGRD -- u-component of wind [m/s] -- 70000[Pa] ISBL="Isobaric surface"',  # UNUSED
'VGRD -- v-component of wind [m/s] -- 70000[Pa] ISBL="Isobaric surface"',  # UNUSED
'ABSV -- Absolute vorticity [1/s] -- 70000[Pa] ISBL="Isobaric surface"',  # UNUSED

'HGT -- Geopotential height [gpm] -- 50000[Pa] ISBL="Isobaric surface"',  # UNUSED
'TMP -- Temperature [C] -- 50000[Pa] ISBL="Isobaric surface"',  # UNUSED
'RH -- Relative humidity [%] -- 50000[Pa] ISBL="Isobaric surface"',  # UNUSED
'VVEL -- Vertical velocity (pressure) [Pa/s] -- 50000[Pa] ISBL="Isobaric surface"',  # UNUSED
'UGRD -- u-component of wind [m/s] -- 50000[Pa] ISBL="Isobaric surface"',  # UNUSED
'VGRD -- v-component of wind [m/s] -- 50000[Pa] ISBL="Isobaric surface"',  # UNUSED
'ABSV -- Absolute vorticity [1/s] -- 50000[Pa] ISBL="Isobaric surface"',  # UNUSED
'CLWMR -- Cloud mixing ratio [kg/kg] -- 50000[Pa] ISBL="Isobaric surface"',  # UNUSED

'HGT -- Geopotential height [gpm] -- 30000[Pa] ISBL="Isobaric surface"',  # UNUSED
'TMP -- Temperature [C] -- 30000[Pa] ISBL="Isobaric surface"',  # UNUSED
'RH -- Relative humidity [%] -- 30000[Pa] ISBL="Isobaric surface"',  # UNUSED
'VVEL -- Vertical velocity (pressure) [Pa/s] -- 30000[Pa] ISBL="Isobaric surface"',  # UNUSED
'UGRD -- u-component of wind [m/s] -- 30000[Pa] ISBL="Isobaric surface"',  # UNUSED
'VGRD -- v-component of wind [m/s] -- 30000[Pa] ISBL="Isobaric surface"',  # UNUSED
'ABSV -- Absolute vorticity [1/s] -- 30000[Pa] ISBL="Isobaric surface"',  # UNUSED
'CLWMR -- Cloud mixing ratio [kg/kg] -- 30000[Pa] ISBL="Isobaric surface"',  # UNUSED

'HGT -- Geopotential height [gpm] -- 10000[Pa] ISBL="Isobaric surface"',  # UNUSED
'TMP -- Temperature [C] -- 10000[Pa] ISBL="Isobaric surface"',  # UNUSED
'RH -- Relative humidity [%] -- 10000[Pa] ISBL="Isobaric surface"',  # UNUSED
'VVEL -- Vertical velocity (pressure) [Pa/s] -- 10000[Pa] ISBL="Isobaric surface"',  # UNUSED
'UGRD -- u-component of wind [m/s] -- 10000[Pa] ISBL="Isobaric surface"',  # UNUSED
'VGRD -- v-component of wind [m/s] -- 10000[Pa] ISBL="Isobaric surface"',  # UNUSED
'ABSV -- Absolute vorticity [1/s] -- 10000[Pa] ISBL="Isobaric surface"',  # UNUSED
'CLWMR -- Cloud mixing ratio [kg/kg] -- 10000[Pa] ISBL="Isobaric surface"',  # UNUSED
]

operators = { # 'name' : (a, b) -> a + b * value
# https://www.nco.ncep.noaa.gov/pmb/docs/on388/table2.html
'LANDN -- Land-sea coverage (nearest neighbor) [0=sea; 1=land] -- 0[-] SFC="Ground or water surface"' : (0, 1),
'ICEC -- Ice cover [Proportion] -- 0[-] SFC="Ground or water surface"' : (0, 1),
'SUNSD -- SunShine duration [s] -- 0[-] SFC="Ground or water surface"' : (0, 1.0 / 450.0),
'RH -- Relative humidity [%] -- 2[m] HTGL="Specified height level above ground"' : (0, 1.0 / 100.0),  # UNUSED
# seems wonky
#'SNOD -- Snow depth [m] -- 0[-] SFC="Ground or water surface"',  # UNUSED
'UGRD -- u-component of wind [m/s] -- 10[m] HTGL="Specified height level above ground"' : (0.5, 1.0 / 200.0),  # UNUSED
'VGRD -- v-component of wind [m/s] -- 10[m] HTGL="Specified height level above ground"' : (0.5, 1.0 / 200.0),  # UNUSED
'PRES -- Pressure [Pa] -- 0[-] SFC="Ground or water surface"' : (0, 1.0 / 100000.0),
'HGT -- Geopotential height [gpm] -- 0[-] SFC="Ground or water surface"' : (0.01, 1.0 / 10000.0),  # UNUSED
'TMP -- Temperature [C] -- 0[-] SFC="Ground or water surface"' : (0.5, 1.0 / 55.0),  # UNUSED

'HGT -- Geopotential height [gpm] -- 90000[Pa] ISBL="Isobaric surface"' : (-0.5, 1 / 1500.0),  # UNUSED
'TMP -- Temperature [C] -- 90000[Pa] ISBL="Isobaric surface"' : (0.5, 1.0 / 55.0),  # UNUSED
'RH -- Relative humidity [%] -- 90000[Pa] ISBL="Isobaric surface"' : (0, 1.0 / 100.0),  # UNUSED
'VVEL -- Vertical velocity (pressure) [Pa/s] -- 90000[Pa] ISBL="Isobaric surface"' : (1, 1),  # UNUSED
'UGRD -- u-component of wind [m/s] -- 90000[Pa] ISBL="Isobaric surface"' : (0.5, 1.0 / 200.0),  # UNUSED
'VGRD -- v-component of wind [m/s] -- 90000[Pa] ISBL="Isobaric surface"' : (0.5, 1.0 / 200.0),  # UNUSED
'ABSV -- Absolute vorticity [1/s] -- 90000[Pa] ISBL="Isobaric surface"' : (0.5, 100),  # UNUSED
'CLWMR -- Cloud mixing ratio [kg/kg] -- 90000[Pa] ISBL="Isobaric surface"' : (0, 1000.0),  # UNUSED

'HGT -- Geopotential height [gpm] -- 70000[Pa] ISBL="Isobaric surface"' : (-0.9, 1.0 / 2000.0),  # UNUSED
'TMP -- Temperature [C] -- 70000[Pa] ISBL="Isobaric surface"' : (0.5, 1.0 / 55.0),  # UNUSED
'RH -- Relative humidity [%] -- 70000[Pa] ISBL="Isobaric surface"' : (0, 1.0 / 100.0),  # UNUSED
'VVEL -- Vertical velocity (pressure) [Pa/s] -- 70000[Pa] ISBL="Isobaric surface"' : (1, 1),  # UNUSED
'UGRD -- u-component of wind [m/s] -- 70000[Pa] ISBL="Isobaric surface"' : (0.5, 1.0 / 200.0),  # UNUSED
'VGRD -- v-component of wind [m/s] -- 70000[Pa] ISBL="Isobaric surface"' : (0.5, 1.0 / 200.0),  # UNUSED
'ABSV -- Absolute vorticity [1/s] -- 70000[Pa] ISBL="Isobaric surface"' : (0.5, 100.0),  # UNUSED

'HGT -- Geopotential height [gpm] -- 50000[Pa] ISBL="Isobaric surface"' : (-2, 1.0 / 2000.0),  # UNUSED
'TMP -- Temperature [C] -- 50000[Pa] ISBL="Isobaric surface"' : (0.5, 1.0 / 55.0),  # UNUSED
'RH -- Relative humidity [%] -- 50000[Pa] ISBL="Isobaric surface"' : (0, 1.0 / 100.0),  # UNUSED
'VVEL -- Vertical velocity (pressure) [Pa/s] -- 50000[Pa] ISBL="Isobaric surface"' : (1, 1),  # UNUSED
'UGRD -- u-component of wind [m/s] -- 50000[Pa] ISBL="Isobaric surface"' : (0.5, 1.0 / 200.0),  # UNUSED
'VGRD -- v-component of wind [m/s] -- 50000[Pa] ISBL="Isobaric surface"' : (0.5, 1.0 / 200.0),  # UNUSED
'ABSV -- Absolute vorticity [1/s] -- 50000[Pa] ISBL="Isobaric surface"' : (0.5, 100.0),  # UNUSED
'CLWMR -- Cloud mixing ratio [kg/kg] -- 50000[Pa] ISBL="Isobaric surface"' : (0, 100.0),  # UNUSED

'HGT -- Geopotential height [gpm] -- 30000[Pa] ISBL="Isobaric surface"' : (-2.5, 1.0 / 3000.0),  # UNUSED
'TMP -- Temperature [C] -- 30000[Pa] ISBL="Isobaric surface"' : (0.5, 1.0 / 55.0),  # UNUSED
'RH -- Relative humidity [%] -- 30000[Pa] ISBL="Isobaric surface"' : (0, 1.0 / 100.0),  # UNUSED
'VVEL -- Vertical velocity (pressure) [Pa/s] -- 30000[Pa] ISBL="Isobaric surface"' : (1, 1),  # UNUSED
'UGRD -- u-component of wind [m/s] -- 30000[Pa] ISBL="Isobaric surface"' : (0.5, 1.0 / 200.0),  # UNUSED
'VGRD -- v-component of wind [m/s] -- 30000[Pa] ISBL="Isobaric surface"' : (0.5, 1.0 / 200.0),  # UNUSED
'ABSV -- Absolute vorticity [1/s] -- 30000[Pa] ISBL="Isobaric surface"' : (0.5, 100.0),  # UNUSED
'CLWMR -- Cloud mixing ratio [kg/kg] -- 30000[Pa] ISBL="Isobaric surface"' : (0, 100.0),  # UNUSED

'HGT -- Geopotential height [gpm] -- 10000[Pa] ISBL="Isobaric surface"' : (-4.7, 1 / 3000.0),  # UNUSED
'TMP -- Temperature [C] -- 10000[Pa] ISBL="Isobaric surface"' : (0.5, 1.0 / 55.0),  # UNUSED
'RH -- Relative humidity [%] -- 10000[Pa] ISBL="Isobaric surface"' : (0, 1.0 / 100.0),  # UNUSED
'VVEL -- Vertical velocity (pressure) [Pa/s] -- 10000[Pa] ISBL="Isobaric surface"' : (1, 1),  # UNUSED
'UGRD -- u-component of wind [m/s] -- 10000[Pa] ISBL="Isobaric surface"' : (0.5, 1.0 / 200.0),  # UNUSED
'VGRD -- v-component of wind [m/s] -- 10000[Pa] ISBL="Isobaric surface"' : (0.5, 1.0 / 200.0),  # UNUSED
'ABSV -- Absolute vorticity [1/s] -- 10000[Pa] ISBL="Isobaric surface"' : (0.5, 100.0),  # UNUSED
'CLWMR -- Cloud mixing ratio [kg/kg] -- 10000[Pa] ISBL="Isobaric surface"' : (0, 1000.0),  # UNUSED
}