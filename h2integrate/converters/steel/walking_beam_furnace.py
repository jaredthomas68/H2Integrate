from h2integrate.core.model_baseclasses import CostModelBaseClass, PerformanceModelBaseClass


class NaturalGasWalkingBeamFurnacePerformanceModel(PerformanceModelBaseClass):
    _time_step_bounds = (
        300,
        3600,
    )  # (min, max) time step lengths (in seconds) compatible with this model

    def initialize(self):
        super().initialize()
        self.commodity = "steel"
        self.commodity_amount_units = "t"
        self.commodity_rate_units = "t/h"

    def setup(self):
        super().setup()
        n_timesteps = self.options["plant_config"]["plant"]["simulation"]["n_timesteps"]
        self.add_input("electricity_in", val=0.0, shape=n_timesteps, units="kW")
        self.add_input("natural_gas", val=0.0, shape=n_timesteps, units="kg/h")

    def compute(self, inputs, outputs):
        """
        Computation for the walking beam furnace component.

        """


class NaturalGasWalkingBeamFurnaceCostModel(CostModelBaseClass):
    _time_step_bounds = (
        300,
        3600,
    )  # (min, max) time step lengths (in seconds) compatible with this model

    def setup(self):
        # Inputs for cost model configuration
        super().setup()
        self.add_input("plant_capacity_mtpy", val=0.0, units="t/year", desc="Annual plant capacity")
        self.add_input("plant_capacity_factor", val=0.0, units=None, desc="Capacity factor")
        self.add_input("LCOH", val=0.0, units="USD/kg", desc="Levelized cost of hydrogen")
        self.add_input(
            "electricity_cost", val=0.0, units="USD/(MW*h)", desc="Levelized cost of electricity"
        )

    def compute(self, inputs, outputs):
        return
