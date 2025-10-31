from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import openmdao.api as om
from attrs import field, define

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.validators import gt_zero, range_val


@define
class SimpleGenericStorageConfig(BaseConfig):
    commodity_name: str = field()
    commodity_units: str = field()


class SimpleGenericStorage(om.ExplicitComponent):
    """
    Simple generic storage model.
    """

    def initialize(self):
        self.options.declare("tech_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("driver_config", types=dict)

    def setup(self):
        n_timesteps = self.options["plant_config"]["plant"]["simulation"]["n_timesteps"]
        self.config = SimpleGenericStorageConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            strict=False,
        )
        commodity_name = self.config.commodity_name
        commodity_units = self.config.commodity_units
        self.add_input(f"{commodity_name}_in", val=0.0, shape=n_timesteps, units=commodity_units)

    def compute(self, inputs, outputs):
        pass


@dataclass
class BatteryOutputs:
    # I: Sequence
    # P: Sequence
    # Q: Sequence
    SOC: Sequence
    # T_batt: Sequence
    # gen: Sequence
    # n_cycles: Sequence
    # P_chargeable: Sequence
    # P_dischargeable: Sequence
    # unmet_demand: list[float]
    # unused_commodity: list[float]


@define
class SimpleGenericStoragePyomoConfig(BaseConfig):
    commodity_name: str = field()
    commodity_units: str = field()
    max_capacity: float = field(validator=gt_zero)
    max_charge_rate: float = field(validator=gt_zero)
    min_charge_percent: float = field(validator=range_val(0, 1))
    max_charge_percent: float = field(validator=range_val(0, 1))
    init_charge_percent: float = field(validator=range_val(0, 1))
    n_control_window: int = field(validator=gt_zero, default=24)
    n_horizon_window: int = field(validator=gt_zero, default=48)


class SimpleGenericStoragePyomo(om.ExplicitComponent):
    """
    Simple generic storage model.
    """

    def initialize(self):
        self.options.declare("tech_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("driver_config", types=dict)

    def setup(self):
        n_timesteps = self.options["plant_config"]["plant"]["simulation"]["n_timesteps"]
        self.config = SimpleGenericStoragePyomoConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            strict=False,
        )
        self.commodity_name = self.config.commodity_name
        commodity_units = self.config.commodity_units
        self.add_input(
            f"{self.commodity_name}_in", val=0.0, shape=n_timesteps, units=commodity_units
        )

        self.add_output(
            f"{self.commodity_name}_out",
            val=0.0,
            copy_shape=f"{self.commodity_name}_in",
            units=f"{commodity_units}",
            desc="Total electricity out of Battery",
        )

        self.add_output(
            "SOC",
            val=0.0,
            copy_shape=f"{self.commodity_name}_in",
            units="percent",
            desc="State of charge of Battery",
        )

        self.add_output(
            f"storage_{self.commodity_name}_discharge",
            val=0.0,
            copy_shape=f"{self.commodity_name}_in",
            units=f"{commodity_units}",
            desc=f"{self.commodity_name} output from storage only",
        )

        self.add_input(
            "max_charge_rate",
            val=self.config.max_charge_rate,
            units=f"{commodity_units}",
            desc="Storage charge rate",
        )

        self.add_input(
            "storage_capacity",
            val=self.config.max_capacity,
            units=f"{commodity_units}*h",
            desc="Storage capacity",
        )

        self.add_input(
            f"{self.commodity_name}_demand",
            val=0.0,
            copy_shape=f"{self.commodity_name}_in",
            units=f"{commodity_units}",
            desc=f"{self.commodity_name} demand",
        )

        self.add_output(
            f"unmet_{self.commodity_name}_demand_out",
            val=0.0,
            copy_shape=f"{self.commodity_name}_in",
            units=f"{commodity_units}",
            desc=f"Unmet {self.commodity_name} demand",
        )

        self.add_output(
            f"unused_{self.commodity_name}_out",
            val=0.0,
            copy_shape=f"{self.commodity_name}_in",
            units=f"{commodity_units}",
            desc=f"Unused generated {self.commodity_name}",
        )

        n_timesteps = int(
            self.options["plant_config"]["plant"]["simulation"]["n_timesteps"]
        )  # self.config.n_timesteps
        self.dt_hr = int(self.options["plant_config"]["plant"]["simulation"]["dt"]) / (
            60**2
        )  # convert from seconds to hours

        self.outputs = BatteryOutputs()
        self.outputs.SOC = []
        self.outputs.SOC.append(self.config.init_charge_percent / 100.0)

        # create inputs for pyomo control model
        if "tech_to_dispatch_connections" in self.options["plant_config"]:
            # get technology group name
            # TODO: The split below seems brittle
            self.tech_group_name = self.pathname.split(".")
            for _source_tech, intended_dispatch_tech in self.options["plant_config"][
                "tech_to_dispatch_connections"
            ]:
                if any(intended_dispatch_tech in name for name in self.tech_group_name):
                    self.add_discrete_input("pyomo_dispatch_solver", val=dummy_function)
                    break

        self.unmet_demand = 0.0
        self.unused_commodity = 0.0

    def compute(self, inputs, outputs, discrete_inputs=[], discrete_outputs=[]):
        if "pyomo_dispatch_solver" in discrete_inputs:
            # Simulate the battery with provided dispatch inputs
            dispatch = discrete_inputs["pyomo_dispatch_solver"]
            kwargs = {
                "time_step_duration": self.dt_hr,
                "control_variable": self.config.control_variable,
            }
            (
                total_commodity_out,
                storage_commodity_out,
                unmet_demand,
                unused_commodity,
                soc,
            ) = dispatch(self.simulate, kwargs, inputs)

            outputs[f"unmet_{self.commodity_name}_demand_out"] = unmet_demand
            outputs[f"unused_{self.commodity_name}_out"] = unused_commodity
            outputs[f"storage_{self.commodity_name}_discharge"] = storage_commodity_out
            outputs[f"{self.commodity_name}_out"] = total_commodity_out
            outputs["SOC"] = soc

        else:
            pass

    def simulate(
        self,
        storage_dispatch_commands: list,
        time_step_duration: list,
        control_variable: str,
        sim_start_index: int = 0,
    ):
        """Run the dispatch model over a control window.

        Applies a sequence of dispatch commands (positive = discharge, negative = charge)
        one timestep at a time. Each command is clipped to allowable instantaneous
        charge / discharge limits derived from:
          1. Rated power (config.max_charge_rate)
          2. PySAM internal estimates (P_chargeable / P_dischargeable)
          3. Remaining energy headroom vs. SOC bounds

        The method updates internal rolling arrays in self.outputs in-place using
        sim_start_index as an offset (enabling sliding / receding horizon logic).

        The simulate method is much of what would normally be in the compute() method
        of a component, but is separated into its own function here to allow the dispatch()
        method to manage calls to the performance model.

        Args:
            storage_dispatch_commands : Sequence[float]
                Commanded power per timestep (kW). Negative = charge, positive = discharge.
                Length should be = config.n_control_window.
            time_step_duration : float | Sequence[float]
                Timestep duration in hours. Scalar applied uniformly or sequence matching
                len(storage_dispatch_commands).
            control_variable : str
                PySAM control input to set each step ("input_power" or "input_current").
            sim_start_index : int, optional
                Starting index for writing into persistent output arrays (default 0).

        Returns:
            tuple[np.ndarray, np.ndarray]
                (battery_power_kW, soc_percent)
                battery_power_kW : array of PySAM P values (kW) per timestep
                                    (positive = discharge, negative = charge).
                soc_percent      : array of SOC values (%) per timestep.

        Notes:
            - SOC bounds may still be exceeded slightly due to PySAM internal dynamics.
            - self.outputs.stateful_attributes are updated only if the attribute exists
            in StatePack or StateCell.
            - self.outputs.component_attributes (e.g., unmet_demand) are not modified here;
            they are populated in compute(), unless an external dispatcher manages them.
        """

        # Loop through the provided input power/current (decided by control_variable)
        # self.system_model.value("dt_hr", time_step_duration)

        # initialize outputs
        storage_commodity_out_timesteps = np.zeros(self.config.n_control_window)
        soc_timesteps = np.zeros(self.config.n_control_window)

        # get constant battery parameters needed during all time steps
        soc_max = self.config.max_charge_percent / 100.0
        soc_min = self.config.min_charge_percent / 100.0

        for t, dispatch_command_t in enumerate(storage_dispatch_commands):
            # get storage SOC at time t
            soc = self.outputs.SOC[-1]

            # manually adjust the dispatch command based on SOC
            ## for when battery is withing set bounds
            # according to specs
            max_chargeable_0 = self.config.max_charge_rate
            # according to simulation
            # max_chargeable_1 = np.maximum(0, -self.system_model.value("P_chargeable"))
            # according to soc
            max_chargeable_2 = np.maximum(
                0, (soc_max - soc) * self.config.max_capacity / self.dt_hr
            )
            # compare all versions of max_chargeable
            max_chargeable = np.min([max_chargeable_0, max_chargeable_2])

            # according to specs
            max_dischargeable_0 = self.config.max_charge_rate
            # according to simulation
            # max_dischargeable_1 = np.maximum(0, self.system_model.value("P_dischargeable"))
            # according to soc
            max_dischargeable_2 = np.maximum(
                0, (soc - soc_min) * self.config.max_capacity / self.dt_hr
            )
            # compare all versions of max_dischargeable
            max_dischargeable = np.min([max_dischargeable_0, max_dischargeable_2])

            if dispatch_command_t < -max_chargeable:
                dispatch_command_t = -max_chargeable
            if dispatch_command_t > max_dischargeable:
                dispatch_command_t = max_dischargeable

            # if battery soc is outside the set bounds, discharge battery down to set bounds
            if (soc > soc_max) and dispatch_command_t < 0:  # and (dispatch_command_t <= 0):
                dispatch_command_t = 0.0

            # Set the input variable to the desired value
            # self.system_model.value(control_variable, dispatch_command_t)

            # Simulate the PySAM BatteryStateful model
            # self.system_model.execute(0)
            self.outputs.SOC.append(soc - dispatch_command_t / self.config.max_capacity)

            # save outputs
            storage_commodity_out_timesteps[t] = dispatch_command_t
            soc_timesteps[t] = self.outputs.SOC[-1]

        return storage_commodity_out_timesteps, soc_timesteps


def dummy_function():
    # this function is required for initializing the pyomo control input and nothing else
    pass
