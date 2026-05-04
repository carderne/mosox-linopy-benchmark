import numpy as np
import xarray as xr
import linopy

INTEGER_DIMS = {"YEAR", "MODE_OF_OPERATION", "SEASON", "DAYTYPE", "DAILYTIMEBRACKET"}


def _coerce(dim: str, value: str):
    base_dim = "REGION" if dim == "REGION2" else dim
    return int(value) if base_dim in INTEGER_DIMS else value


def coords_from_data(data: dict) -> dict[str, xr.IndexVariable]:
    coords = {
        dim: xr.IndexVariable(dim, [_coerce(dim, value) for value in values])
        for dim, values in data["sets"].items()
    }
    coords["REGION2"] = xr.IndexVariable("REGION2", list(coords["REGION"].values))
    return coords


def parameter_from_data(
    name: str,
    data: dict,
    param_dims: dict[str, list[str]],
    coords: dict[str, xr.IndexVariable],
) -> xr.DataArray:
    dims = param_dims[name]
    shape = [len(coords[dim]) for dim in dims]
    array = xr.DataArray(
        np.full(shape, data["params"][name]["default"], dtype=float),
        coords={dim: coords[dim] for dim in dims},
        dims=dims,
        name=name,
    )

    for key, value in data["params"][name]["entries"].items():
        selector = {
            dim: _coerce(dim, token) for dim, token in zip(dims, key.split("|"))
        }
        array.loc[selector] = float(value)
    return array


def _any(mask: xr.DataArray) -> bool:
    return bool(mask.any())


def _expand_timeslice(array: xr.DataArray, timeslice: xr.IndexVariable) -> xr.DataArray:
    return array.expand_dims(TIMESLICE=timeslice)


def build_model(data: dict, param_dims: dict[str, list[str]]) -> linopy.Model:
    coords = coords_from_data(data)
    p = {
        name: parameter_from_data(name, data, param_dims, coords) for name in param_dims
    }

    R = coords["REGION"]
    RR = coords["REGION2"]
    L = coords["TIMESLICE"]
    F = coords["FUEL"]
    T = coords["TECHNOLOGY"]
    M = coords["MODE_OF_OPERATION"]
    Y = coords["YEAR"]
    E = coords["EMISSION"]
    S = coords["STORAGE"]
    LS = coords["SEASON"]
    LD = coords["DAYTYPE"]
    LH = coords["DAILYTIMEBRACKET"]

    first_year = int(Y[0])
    last_year = int(Y[-1])
    year_values = xr.DataArray(Y.values, coords={"YEAR": Y}, dims="YEAR")

    discount_rate = p["DiscountRate"]
    discount_factor = (1 + discount_rate) ** (year_values - first_year)
    discount_factor_mid = (1 + discount_rate) ** (year_values - first_year + 0.5)
    discount_factor_storage = (1 + p["DiscountRateStorage"]) ** (
        year_values - first_year
    )

    life = p["OperationalLife"]
    capital_recovery_factor = (1 - (1 + discount_rate) ** (-1)) / (
        1 - (1 + discount_rate) ** (-life)
    )
    pv_annuity = (
        (1 - (1 + discount_rate) ** (-life)) * (1 + discount_rate) / discount_rate
    )
    capital_multiplier = capital_recovery_factor * pv_annuity
    years_remaining = last_year - year_values + 1
    salvage_fraction = xr.where(
        (year_values + life - 1) > last_year,
        1
        - (
            ((1 + discount_rate) ** years_remaining - 1)
            / ((1 + discount_rate) ** life - 1)
        ),
        0,
    )
    salvage_discount = (1 + discount_rate) ** (1 + last_year - first_year)
    storage_salvage_discount = (1 + p["DiscountRateStorage"]) ** (
        1 + last_year - first_year
    )

    model = linopy.Model()

    def add(
        name: str, coords_: list[xr.IndexVariable], lower=0, upper=np.inf, integer=False
    ):
        return model.add_variables(
            lower=lower,
            upper=upper,
            coords=coords_,  # ty:ignore[invalid-argument-type]
            name=name,
            integer=integer,
        )

    # Demands
    rate_of_demand = add("RateOfDemand", [R, L, F, Y])
    demand = add("Demand", [R, L, F, Y])

    # Storage
    RSSDDY = [R, S, LS, LD, LH, Y]
    RSY = [R, S, Y]
    rate_storage_charge = add("RateOfStorageCharge", RSSDDY, lower=-np.inf)
    rate_storage_discharge = add("RateOfStorageDischarge", RSSDDY, lower=-np.inf)
    net_charge_year = add("NetChargeWithinYear", RSSDDY, lower=-np.inf)
    net_charge_day = add("NetChargeWithinDay", RSSDDY, lower=-np.inf)
    storage_year_start = add("StorageLevelYearStart", RSY)
    storage_year_finish = add("StorageLevelYearFinish", RSY)
    storage_season_start = add("StorageLevelSeasonStart", [R, S, LS, Y])
    storage_day_start = add("StorageLevelDayTypeStart", [R, S, LS, LD, Y])
    storage_day_finish = add("StorageLevelDayTypeFinish", [R, S, LS, LD, Y])
    storage_lower = add("StorageLowerLimit", RSY)
    storage_upper = add("StorageUpperLimit", RSY)
    accumulated_storage = add("AccumulatedNewStorageCapacity", RSY)
    new_storage = add("NewStorageCapacity", RSY)
    capital_storage = add("CapitalInvestmentStorage", RSY)
    discounted_capital_storage = add("DiscountedCapitalInvestmentStorage", RSY)
    salvage_storage = add("SalvageValueStorage", RSY)
    discounted_salvage_storage = add("DiscountedSalvageValueStorage", RSY)
    total_discounted_storage_cost = add("TotalDiscountedStorageCost", RSY)

    # Capacity
    number_of_new_units = add("NumberOfNewTechnologyUnits", [R, T, Y])
    new_capacity = add("NewCapacity", [R, T, Y])
    accumulated_capacity = add("AccumulatedNewCapacity", [R, T, Y])
    total_capacity = add("TotalCapacityAnnual", [R, T, Y])

    # Activity and energy accounting
    rate = add("RateOfActivity", [R, L, T, M, Y])
    rate_total_activity = add("RateOfTotalActivity", [R, T, L, Y])
    total_annual_activity = add("TotalTechnologyAnnualActivity", [R, T, Y])
    total_annual_activity_by_mode = add(
        "TotalAnnualTechnologyActivityByMode", [R, T, M, Y]
    )
    total_model_activity = add(
        "TotalTechnologyModelPeriodActivity", [R, T], lower=-np.inf
    )
    rate_prod_by_mode = add("RateOfProductionByTechnologyByMode", [R, L, T, M, F, Y])
    rate_prod_by_tech = add("RateOfProductionByTechnology", [R, L, T, F, Y])
    prod_by_tech = add("ProductionByTechnology", [R, L, T, F, Y])
    prod_by_tech_annual = add("ProductionByTechnologyAnnual", [R, T, F, Y])
    rate_prod = add("RateOfProduction", [R, L, F, Y])
    production = add("Production", [R, L, F, Y])
    rate_use_by_mode = add("RateOfUseByTechnologyByMode", [R, L, T, M, F, Y])
    rate_use_by_tech = add("RateOfUseByTechnology", [R, L, T, F, Y])
    use_by_tech_annual = add("UseByTechnologyAnnual", [R, T, F, Y])
    rate_use = add("RateOfUse", [R, L, F, Y])
    use_by_tech = add("UseByTechnology", [R, L, T, F, Y])
    use = add("Use", [R, L, F, Y])
    trade = add("Trade", [R, RR, L, F, Y], lower=-np.inf)
    trade_annual = add("TradeAnnual", [R, RR, F, Y], lower=-np.inf)
    production_annual = add("ProductionAnnual", [R, F, Y])
    use_annual = add("UseAnnual", [R, F, Y])

    # Costs
    capital_investment = add("CapitalInvestment", [R, T, Y])
    discounted_capital_investment = add("DiscountedCapitalInvestment", [R, T, Y])
    salvage_value = add("SalvageValue", [R, T, Y])
    discounted_salvage_value = add("DiscountedSalvageValue", [R, T, Y])
    operating_cost = add("OperatingCost", [R, T, Y])
    discounted_operating_cost = add("DiscountedOperatingCost", [R, T, Y])
    annual_variable_cost = add("AnnualVariableOperatingCost", [R, T, Y])
    annual_fixed_cost = add("AnnualFixedOperatingCost", [R, T, Y])
    total_discounted_cost_by_tech = add("TotalDiscountedCostByTechnology", [R, T, Y])
    total_discounted_cost = add("TotalDiscountedCost", [R, Y])
    model_period_cost = add("ModelPeriodCostByRegion", [R])

    # Reserve margin and RE target
    reserve_capacity = add("TotalCapacityInReserveMargin", [R, Y])
    reserve_demand = add("DemandNeedingReserveMargin", [R, L, Y])
    total_re_production = add("TotalREProductionAnnual", [R, Y], lower=-np.inf)
    re_target_fuel_production = add(
        "RETotalProductionOfTargetFuelAnnual", [R, Y], lower=-np.inf
    )

    # Emissions
    annual_emission_by_mode = add("AnnualTechnologyEmissionByMode", [R, T, E, M, Y])
    annual_emission_by_tech = add("AnnualTechnologyEmission", [R, T, E, Y])
    emission_penalty_by_emission = add(
        "AnnualTechnologyEmissionPenaltyByEmission", [R, T, E, Y]
    )
    annual_emission_penalty = add("AnnualTechnologyEmissionsPenalty", [R, T, Y])
    discounted_emission_penalty = add("DiscountedTechnologyEmissionsPenalty", [R, T, Y])
    annual_emissions = add("AnnualEmissions", [R, E, Y])
    model_period_emissions = add("ModelPeriodEmissions", [R, E])

    # Demand equations
    model.add_constraints(
        rate_of_demand
        == p["SpecifiedAnnualDemand"] * p["SpecifiedDemandProfile"] / p["YearSplit"],
        name="EQ_SpecifiedDemand",
        mask=p["SpecifiedAnnualDemand"] != 0,
    )

    # Capacity adequacy A/B
    accumulated_capacity_expr = None
    for investment_year in Y.values:
        age = year_values - int(investment_year)
        active = ((age >= 0) & (age < life)).astype(float)
        term = new_capacity.sel(YEAR=int(investment_year)) * active
        accumulated_capacity_expr = (
            term
            if accumulated_capacity_expr is None
            else accumulated_capacity_expr + term
        )
    model.add_constraints(
        accumulated_capacity == accumulated_capacity_expr, name="CAa1_TotalNewCapacity"
    )
    model.add_constraints(
        accumulated_capacity + p["ResidualCapacity"] == total_capacity,
        name="CAa2_TotalAnnualCapacity",
    )
    model.add_constraints(
        rate.sum("MODE_OF_OPERATION") == rate_total_activity,
        name="CAa3_TotalActivityOfEachTechnology",
    )
    model.add_constraints(
        rate_total_activity
        <= total_capacity * p["CapacityFactor"] * p["CapacityToActivityUnit"],
        name="CAa4_Constraint_Capacity",
    )
    if _any(p["CapacityOfOneTechnologyUnit"] != 0):
        model.add_constraints(
            p["CapacityOfOneTechnologyUnit"] * number_of_new_units == new_capacity,
            name="CAa5_TotalNewCapacity",
            mask=p["CapacityOfOneTechnologyUnit"] != 0,
        )
    if _any(p["AvailabilityFactor"] < 1):
        model.add_constraints(
            (rate_total_activity * p["YearSplit"]).sum("TIMESLICE")
            <= (total_capacity * p["CapacityFactor"] * p["YearSplit"]).sum("TIMESLICE")
            * p["AvailabilityFactor"]
            * p["CapacityToActivityUnit"],
            name="CAb1_PlannedMaintenance",
            mask=p["AvailabilityFactor"] < 1,
        )

    # Energy balance A
    output_nonzero = (p["OutputActivityRatio"] != 0).astype(float)
    input_nonzero = (p["InputActivityRatio"] != 0).astype(float)
    output_exists = (
        p["OutputActivityRatio"].sum(["TECHNOLOGY", "MODE_OF_OPERATION"]) != 0
    )
    input_exists = p["InputActivityRatio"].sum(["TECHNOLOGY", "MODE_OF_OPERATION"]) != 0

    model.add_constraints(
        rate * p["OutputActivityRatio"] == rate_prod_by_mode,
        name="EBa1_RateOfFuelProduction1",
        mask=p["OutputActivityRatio"] != 0,
    )
    model.add_constraints(
        (rate_prod_by_mode * output_nonzero).sum("MODE_OF_OPERATION")
        == rate_prod_by_tech,
        name="EBa2_RateOfFuelProduction2",
    )
    model.add_constraints(
        rate_prod_by_tech.sum("TECHNOLOGY") == rate_prod,
        name="EBa3_RateOfFuelProduction3",
        mask=output_exists,
    )
    model.add_constraints(
        rate * p["InputActivityRatio"] == rate_use_by_mode,
        name="EBa4_RateOfFuelUse1",
        mask=p["InputActivityRatio"] != 0,
    )
    model.add_constraints(
        (rate_use_by_mode * input_nonzero).sum("MODE_OF_OPERATION") == rate_use_by_tech,
        name="EBa5_RateOfFuelUse2",
        mask=p["InputActivityRatio"].sum("MODE_OF_OPERATION") != 0,
    )
    model.add_constraints(
        rate_use_by_tech.sum("TECHNOLOGY") == rate_use,
        name="EBa6_RateOfFuelUse3",
        mask=input_exists,
    )
    model.add_constraints(
        rate_prod * p["YearSplit"] == production,
        name="EBa7_EnergyBalanceEachTS1",
        mask=output_exists,
    )
    model.add_constraints(
        rate_use * p["YearSplit"] == use,
        name="EBa8_EnergyBalanceEachTS2",
        mask=input_exists,
    )
    model.add_constraints(
        rate_of_demand * p["YearSplit"] == demand,
        name="EBa9_EnergyBalanceEachTS3",
        mask=p["SpecifiedAnnualDemand"] != 0,
    )
    if _any(p["TradeRoute"] != 0):
        reverse_trade = trade.rename(REGION="REGION2", REGION2="REGION")
        model.add_constraints(
            trade == -reverse_trade,
            name="EBa10_EnergyBalanceEachTS4",
            mask=p["TradeRoute"] != 0,
        )
    model.add_constraints(
        production >= demand + use + (trade * p["TradeRoute"]).sum("REGION2"),
        name="EBa11_EnergyBalanceEachTS5",
    )

    # Energy balance B and accounting
    model.add_constraints(
        production.sum("TIMESLICE") == production_annual,
        name="EBb1_EnergyBalanceEachYear1",
    )
    model.add_constraints(
        use.sum("TIMESLICE") == use_annual, name="EBb2_EnergyBalanceEachYear2"
    )
    model.add_constraints(
        trade.sum("TIMESLICE") == trade_annual, name="EBb3_EnergyBalanceEachYear3"
    )
    model.add_constraints(
        production_annual
        >= use_annual
        + (trade_annual * p["TradeRoute"]).sum("REGION2")
        + p["AccumulatedAnnualDemand"],
        name="EBb4_EnergyBalanceEachYear4",
    )
    model.add_constraints(
        rate_prod_by_tech * p["YearSplit"] == prod_by_tech,
        name="Acc1_FuelProductionByTechnology",
    )
    model.add_constraints(
        rate_use_by_tech * p["YearSplit"] == use_by_tech,
        name="Acc2_FuelUseByTechnology",
    )
    model.add_constraints(
        (rate * p["YearSplit"]).sum("TIMESLICE") == total_annual_activity_by_mode,
        name="Acc3_AverageAnnualRateOfActivity",
    )
    model.add_constraints(
        total_discounted_cost.sum("YEAR") == model_period_cost,
        name="Acc4_ModelPeriodCostByRegion",
    )

    # Storage equations
    model.add_constraints(
        (
            rate
            * p["TechnologyToStorage"]
            * p["Conversionls"]
            * p["Conversionld"]
            * p["Conversionlh"]
        ).sum(["TECHNOLOGY", "MODE_OF_OPERATION", "TIMESLICE"])
        == rate_storage_charge,
        name="S1_RateOfStorageCharge",
    )
    model.add_constraints(
        (
            rate
            * p["TechnologyFromStorage"]
            * p["Conversionls"]
            * p["Conversionld"]
            * p["Conversionlh"]
        ).sum(["TECHNOLOGY", "MODE_OF_OPERATION", "TIMESLICE"])
        == rate_storage_discharge,
        name="S2_RateOfStorageDischarge",
    )
    model.add_constraints(
        (
            (rate_storage_charge - rate_storage_discharge)
            * p["YearSplit"]
            * p["Conversionls"]
            * p["Conversionld"]
            * p["Conversionlh"]
        ).sum("TIMESLICE")
        == net_charge_year,
        name="S3_NetChargeWithinYear",
    )
    model.add_constraints(
        (rate_storage_charge - rate_storage_discharge) * p["DaySplit"]
        == net_charge_day,
        name="S4_NetChargeWithinDay",
    )

    for y in Y.values:
        y = int(y)
        if y == first_year:
            model.add_constraints(
                storage_year_start.sel(YEAR=y) == p["StorageLevelStart"],
                name=f"S5_StorageLevelYearStart_{y}",
            )
        else:
            model.add_constraints(
                storage_year_start.sel(YEAR=y - 1)
                + net_charge_year.sel(YEAR=y - 1).sum(
                    ["SEASON", "DAYTYPE", "DAILYTIMEBRACKET"]
                )
                == storage_year_start.sel(YEAR=y),
                name=f"S6_StorageLevelYearStart_{y}",
            )

        if y < last_year:
            model.add_constraints(
                storage_year_start.sel(YEAR=y + 1) == storage_year_finish.sel(YEAR=y),
                name=f"S7_StorageLevelYearFinish_{y}",
            )
        else:
            model.add_constraints(
                storage_year_start.sel(YEAR=y)
                + net_charge_year.sel(YEAR=y).sum(
                    ["SEASON", "DAYTYPE", "DAILYTIMEBRACKET"]
                )
                == storage_year_finish.sel(YEAR=y),
                name=f"S8_StorageLevelYearFinish_{y}",
            )

    season_vals = [int(x) for x in LS.values]
    daytype_vals = [int(x) for x in LD.values]
    for ls in season_vals:
        if ls == min(season_vals):
            model.add_constraints(
                storage_year_start == storage_season_start.sel(SEASON=ls),
                name=f"S9_StorageLevelSeasonStart_{ls}",
            )
        else:
            model.add_constraints(
                storage_season_start.sel(SEASON=ls - 1)
                + net_charge_year.sel(SEASON=ls - 1).sum(
                    ["DAYTYPE", "DAILYTIMEBRACKET"]
                )
                == storage_season_start.sel(SEASON=ls),
                name=f"S10_StorageLevelSeasonStart_{ls}",
            )
        for ld in daytype_vals:
            if ld == min(daytype_vals):
                model.add_constraints(
                    storage_season_start.sel(SEASON=ls)
                    == storage_day_start.sel(SEASON=ls, DAYTYPE=ld),
                    name=f"S11_StorageLevelDayTypeStart_{ls}_{ld}",
                )
            else:
                model.add_constraints(
                    storage_day_start.sel(SEASON=ls, DAYTYPE=ld - 1)
                    + (
                        net_charge_day.sel(SEASON=ls, DAYTYPE=ld - 1)
                        * p["DaysInDayType"].sel(SEASON=ls, DAYTYPE=ld - 1)
                    ).sum("DAILYTIMEBRACKET")
                    == storage_day_start.sel(SEASON=ls, DAYTYPE=ld),
                    name=f"S12_StorageLevelDayTypeStart_{ls}_{ld}",
                )

            if ls == max(season_vals) and ld == max(daytype_vals):
                model.add_constraints(
                    storage_year_finish
                    == storage_day_finish.sel(SEASON=ls, DAYTYPE=ld),
                    name=f"S13_StorageLevelDayTypeFinish_{ls}_{ld}",
                )
            elif ld == max(daytype_vals):
                model.add_constraints(
                    storage_season_start.sel(SEASON=ls + 1)
                    == storage_day_finish.sel(SEASON=ls, DAYTYPE=ld),
                    name=f"S14_StorageLevelDayTypeFinish_{ls}_{ld}",
                )
            else:
                model.add_constraints(
                    storage_day_finish.sel(SEASON=ls, DAYTYPE=ld + 1)
                    - (
                        net_charge_day.sel(SEASON=ls, DAYTYPE=ld + 1)
                        * p["DaysInDayType"].sel(SEASON=ls, DAYTYPE=ld + 1)
                    ).sum("DAILYTIMEBRACKET")
                    == storage_day_finish.sel(SEASON=ls, DAYTYPE=ld),
                    name=f"S15_StorageLevelDayTypeFinish_{ls}_{ld}",
                )

    # Storage constraints
    for lh in [int(x) for x in LH.values]:
        before = [int(x) for x in LH.values if lh - int(x) > 0]
        after = [int(x) for x in LH.values if lh - int(x) < 0]
        sc1_level = storage_day_start + (
            net_charge_day.sel(DAILYTIMEBRACKET=before).sum("DAILYTIMEBRACKET")
            if before
            else 0
        )
        model.add_constraints(sc1_level >= storage_lower, name=f"SC1_Lower_{lh}")
        model.add_constraints(sc1_level <= storage_upper, name=f"SC1_Upper_{lh}")
        sc3_level = storage_day_finish - (
            net_charge_day.sel(DAILYTIMEBRACKET=after).sum("DAILYTIMEBRACKET")
            if after
            else 0
        )
        model.add_constraints(sc3_level >= storage_lower, name=f"SC3_Lower_{lh}")
        model.add_constraints(sc3_level <= storage_upper, name=f"SC3_Upper_{lh}")
    model.add_constraints(
        rate_storage_charge <= p["StorageMaxChargeRate"], name="SC5_MaxChargeConstraint"
    )
    model.add_constraints(
        rate_storage_discharge <= p["StorageMaxDischargeRate"],
        name="SC6_MaxDischargeConstraint",
    )

    # Storage investments
    model.add_constraints(
        accumulated_storage + p["ResidualStorageCapacity"] == storage_upper,
        name="SI1_StorageUpperLimit",
    )
    model.add_constraints(
        p["MinStorageCharge"] * storage_upper == storage_lower,
        name="SI2_StorageLowerLimit",
    )
    storage_life = p["OperationalLifeStorage"]
    accumulated_storage_expr = None
    for investment_year in Y.values:
        age = year_values - int(investment_year)
        active = ((age >= 0) & (age < storage_life)).astype(float)
        term = new_storage.sel(YEAR=int(investment_year)) * active
        accumulated_storage_expr = (
            term
            if accumulated_storage_expr is None
            else accumulated_storage_expr + term
        )
    model.add_constraints(
        accumulated_storage_expr == accumulated_storage,  # ty:ignore[invalid-argument-type]
        name="SI3_TotalNewStorage",
    )
    model.add_constraints(
        p["CapitalCostStorage"] * new_storage == capital_storage,
        name="SI4_UndiscountedCapitalInvestmentStorage",
    )
    model.add_constraints(
        capital_storage / discount_factor_storage == discounted_capital_storage,
        name="SI5_DiscountingCapitalInvestmentStorage",
    )
    storage_end_before_horizon = (year_values + storage_life - 1) <= last_year
    model.add_constraints(
        salvage_storage == 0,
        name="SI6_SalvageValueStorageAtEndOfPeriod1",
        mask=storage_end_before_horizon,
    )
    storage_after_horizon = (year_values + storage_life - 1) > last_year
    mask_si7 = (
        (p["DepreciationMethod"] == 1)
        & storage_after_horizon
        & (p["DiscountRateStorage"] == 0)
    ) | ((p["DepreciationMethod"] == 2) & storage_after_horizon)
    if _any(mask_si7):
        model.add_constraints(
            capital_storage * (1 - (last_year - year_values + 1) / storage_life)
            == salvage_storage,
            name="SI7_SalvageValueStorageAtEndOfPeriod2",
            mask=mask_si7,
        )
    mask_si8 = (
        (p["DepreciationMethod"] == 1)
        & storage_after_horizon
        & (p["DiscountRateStorage"] > 0)
    )
    if _any(mask_si8):
        model.add_constraints(
            capital_storage
            * (
                1
                - (
                    (
                        (1 + p["DiscountRateStorage"]) ** (last_year - year_values + 1)
                        - 1
                    )
                    / ((1 + p["DiscountRateStorage"]) ** storage_life - 1)
                )
            )
            == salvage_storage,
            name="SI8_SalvageValueStorageAtEndOfPeriod3",
            mask=mask_si8,
        )
    model.add_constraints(
        salvage_storage / storage_salvage_discount == discounted_salvage_storage,
        name="SI9_SalvageValueStorageDiscountedToStartYear",
    )
    model.add_constraints(
        discounted_capital_storage - discounted_salvage_storage
        == total_discounted_storage_cost,
        name="SI10_TotalDiscountedCostByStorage",
    )

    # Capital, salvage, operating and total discounted costs
    model.add_constraints(
        p["CapitalCost"] * new_capacity * capital_multiplier == capital_investment,
        name="CC1_UndiscountedCapitalInvestment",
    )
    model.add_constraints(
        capital_investment / discount_factor == discounted_capital_investment,
        name="CC2_DiscountingCapitalInvestment",
    )

    end_after_horizon = (year_values + life - 1) > last_year
    mask_sv1 = (p["DepreciationMethod"] == 1) & end_after_horizon & (discount_rate > 0)
    model.add_constraints(
        salvage_value
        == p["CapitalCost"] * new_capacity * capital_multiplier * salvage_fraction,
        name="SV1_SalvageValueAtEndOfPeriod1",
        mask=mask_sv1,
    )
    mask_sv2 = (
        (p["DepreciationMethod"] == 1) & end_after_horizon & (discount_rate == 0)
    ) | ((p["DepreciationMethod"] == 2) & end_after_horizon)
    if _any(mask_sv2):
        model.add_constraints(
            salvage_value
            == p["CapitalCost"]
            * new_capacity
            * capital_multiplier
            * (1 - (last_year - year_values + 1) / life),
            name="SV2_SalvageValueAtEndOfPeriod2",
            mask=mask_sv2,
        )
    model.add_constraints(
        salvage_value == 0,
        name="SV3_SalvageValueAtEndOfPeriod3",
        mask=~end_after_horizon,
    )
    model.add_constraints(
        discounted_salvage_value == salvage_value / salvage_discount,
        name="SV4_SalvageValueDiscountedToStartYear",
    )

    variable_cost_expr = (total_annual_activity_by_mode * p["VariableCost"]).sum(
        "MODE_OF_OPERATION"
    )
    oc1_mask = p["VariableCost"].sum("MODE_OF_OPERATION") != 0
    model.add_constraints(
        _expand_timeslice(  # ty:ignore[invalid-argument-type]
            variable_cost_expr,
            L,
        )
        == _expand_timeslice(annual_variable_cost, L),
        name="OC1_OperatingCostsVariable",
        mask=_expand_timeslice(oc1_mask, L),
    )
    model.add_constraints(
        total_capacity * p["FixedCost"] == annual_fixed_cost,
        name="OC2_OperatingCostsFixedAnnual",
    )
    model.add_constraints(
        annual_fixed_cost + annual_variable_cost == operating_cost,
        name="OC3_OperatingCostsTotalAnnual",
    )
    model.add_constraints(
        operating_cost / discount_factor_mid == discounted_operating_cost,
        name="OC4_DiscountedOperatingCostsTotalAnnual",
    )
    model.add_constraints(
        discounted_operating_cost
        + discounted_capital_investment
        + discounted_emission_penalty
        - discounted_salvage_value
        == total_discounted_cost_by_tech,
        name="TDC1_TotalDiscountedCostByTechnology",
    )
    model.add_constraints(
        total_discounted_cost_by_tech.sum("TECHNOLOGY")
        + total_discounted_storage_cost.sum("STORAGE")
        == total_discounted_cost,
        name="TDC2_TotalDiscountedCost",
    )

    # Capacity and activity limits
    model.add_constraints(
        total_capacity <= p["TotalAnnualMaxCapacity"],
        name="TCC1_TotalAnnualMaxCapacityConstraint",
        mask=p["TotalAnnualMaxCapacity"] != -1,
    )
    model.add_constraints(
        total_capacity >= p["TotalAnnualMinCapacity"],
        name="TCC2_TotalAnnualMinCapacityConstraint",
        mask=p["TotalAnnualMinCapacity"] > 0,
    )
    model.add_constraints(
        new_capacity <= p["TotalAnnualMaxCapacityInvestment"],
        name="NCC1_TotalAnnualMaxNewCapacityConstraint",
        mask=p["TotalAnnualMaxCapacityInvestment"] != -1,
    )
    model.add_constraints(
        new_capacity >= p["TotalAnnualMinCapacityInvestment"],
        name="NCC2_TotalAnnualMinNewCapacityConstraint",
        mask=p["TotalAnnualMinCapacityInvestment"] > 0,
    )
    model.add_constraints(
        (rate_total_activity * p["YearSplit"]).sum("TIMESLICE")
        == total_annual_activity,
        name="AAC1_TotalAnnualTechnologyActivity",
    )
    model.add_constraints(
        total_annual_activity <= p["TotalTechnologyAnnualActivityUpperLimit"],
        name="AAC2_TotalAnnualTechnologyActivityUpperLimit",
        mask=p["TotalTechnologyAnnualActivityUpperLimit"] != -1,
    )
    model.add_constraints(
        total_annual_activity >= p["TotalTechnologyAnnualActivityLowerLimit"],
        name="AAC3_TotalAnnualTechnologyActivityLowerLimit",
        mask=p["TotalTechnologyAnnualActivityLowerLimit"] > 0,
    )
    model.add_constraints(
        total_annual_activity.sum("YEAR") == total_model_activity,
        name="TAC1_TotalModelHorizonTechnologyActivity",
    )
    model.add_constraints(
        total_model_activity <= p["TotalTechnologyModelPeriodActivityUpperLimit"],
        name="TAC2_TotalModelHorizonTechnologyActivityUpperLimit",
        mask=p["TotalTechnologyModelPeriodActivityUpperLimit"] != -1,
    )
    model.add_constraints(
        total_model_activity >= p["TotalTechnologyModelPeriodActivityLowerLimit"],
        name="TAC3_TotalModelHorizenTechnologyActivityLowerLimit",
        mask=p["TotalTechnologyModelPeriodActivityLowerLimit"] > 0,
    )

    # Reserve margin
    rm1_expr = (
        total_capacity * p["ReserveMarginTagTechnology"] * p["CapacityToActivityUnit"]
    ).sum("TECHNOLOGY")
    model.add_constraints(
        _expand_timeslice(rm1_expr, L) == _expand_timeslice(reserve_capacity, L),  # ty:ignore[invalid-argument-type]
        name="RM1_ReserveMargin_TechnologiesIncluded_In_Activity_Units",
        mask=_expand_timeslice(p["ReserveMargin"] > 0, L),
    )
    model.add_constraints(
        (rate_prod * p["ReserveMarginTagFuel"]).sum("FUEL") == reserve_demand,
        name="RM2_ReserveMargin_FuelsIncluded",
        mask=p["ReserveMargin"] > 0,
    )
    model.add_constraints(
        reserve_demand * p["ReserveMargin"] <= reserve_capacity,
        name="RM3_ReserveMargin_Constraint",
        mask=p["ReserveMargin"] > 0,
    )

    # RE target
    model.add_constraints(
        prod_by_tech.sum("TIMESLICE") == prod_by_tech_annual,
        name="RE1_FuelProductionByTechnologyAnnual",
    )
    model.add_constraints(
        (prod_by_tech_annual * p["RETagTechnology"]).sum(["TECHNOLOGY", "FUEL"])
        == total_re_production,
        name="RE2_TechIncluded",
    )
    model.add_constraints(
        (rate_prod * p["YearSplit"] * p["RETagFuel"]).sum(["TIMESLICE", "FUEL"])
        == re_target_fuel_production,
        name="RE3_FuelIncluded",
    )
    model.add_constraints(
        p["REMinProductionTarget"] * re_target_fuel_production <= total_re_production,
        name="RE4_EnergyConstraint",
    )
    model.add_constraints(
        (rate_use_by_tech * p["YearSplit"]).sum("TIMESLICE") == use_by_tech_annual,
        name="RE5_FuelUseByTechnologyAnnual",
    )

    # Emissions accounting
    model.add_constraints(
        p["EmissionActivityRatio"] * total_annual_activity_by_mode
        == annual_emission_by_mode,
        name="E1_AnnualEmissionProductionByMode",
        mask=p["EmissionActivityRatio"] != 0,
    )
    model.add_constraints(
        annual_emission_by_mode.sum("MODE_OF_OPERATION") == annual_emission_by_tech,
        name="E2_AnnualEmissionProduction",
    )
    model.add_constraints(
        annual_emission_by_tech * p["EmissionsPenalty"] == emission_penalty_by_emission,
        name="E3_EmissionsPenaltyByTechAndEmission",
        mask=p["EmissionsPenalty"] != 0,
    )
    model.add_constraints(
        emission_penalty_by_emission.sum("EMISSION") == annual_emission_penalty,
        name="E4_EmissionsPenaltyByTechnology",
    )
    model.add_constraints(
        annual_emission_penalty / discount_factor_mid == discounted_emission_penalty,
        name="E5_DiscountedEmissionsPenaltyByTechnology",
    )
    model.add_constraints(
        annual_emission_by_tech.sum("TECHNOLOGY") == annual_emissions,
        name="E6_EmissionsAccounting1",
    )
    model.add_constraints(
        annual_emissions.sum("YEAR")
        == model_period_emissions - p["ModelPeriodExogenousEmission"],
        name="E7_EmissionsAccounting2",
    )
    model.add_constraints(
        annual_emissions + p["AnnualExogenousEmission"] <= p["AnnualEmissionLimit"],
        name="E8_AnnualEmissionsLimit",
        mask=p["AnnualEmissionLimit"] != -1,
    )
    model.add_constraints(
        model_period_emissions <= p["ModelPeriodEmissionLimit"],
        name="E9_ModelPeriodEmissionsLimit",
        mask=p["ModelPeriodEmissionLimit"] != -1,
    )

    model.add_objective(total_discounted_cost.sum(), sense="min")
    return model
