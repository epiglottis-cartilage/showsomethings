pub type Result<T, E: core::fmt::Debug = Box<dyn core::error::Error>> = core::result::Result<T, E>;

// impl core::error::Error for Err {}

#[allow(dead_code)]
pub trait ErrorLogger<T> {
    fn log(self) -> core::result::Result<T, Box<dyn core::error::Error>>;
}
impl<T, E: core::fmt::Debug + core::fmt::Display> ErrorLogger<T> for core::result::Result<T, E> {
    fn log(self) -> Result<T> {
        match self {
            Ok(x) => {
                log::trace!("succ get {}", std::any::type_name::<T>());
                Ok(x)
            }
            Err(e) => {
                log::error!("fail get {}\n{}", std::any::type_name::<T>(), &e);
                #[cfg(debug_assertions)]
                panic!("Aborted due to error: {:?}", e);
                #[cfg(not(debug_assertions))]
                Err(e.into())
            }
        }
    }
}
impl<T> ErrorLogger<T> for core::option::Option<T> {
    fn log(self) -> Result<T> {
        match self {
            Some(x) => {
                log::trace!("succ get {}", std::any::type_name::<T>());
                Ok(x)
            }
            None => {
                log::error!("fail get {}", std::any::type_name::<T>());
                #[cfg(debug_assertions)]
                panic!("None");
                #[cfg(not(debug_assertions))]
                Err(String::from("None").into())
            }
        }
    }
}
